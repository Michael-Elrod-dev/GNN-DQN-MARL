import os
import time
import torch
import numpy as np

from args import Args
from logger import Logger
from collections import deque
from network import GR_QNetwork
from minigrid.world import Agent
from minigrid.environment import MultiGridEnv
from utils import print_info

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size, args):
    """Initialize DDP process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=world_size,
        rank=rank
    )

def cleanup():
    """Clean up DDP process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

def train_ddp(rank, world_size, args):
    setup(rank, world_size, args)
    
    # Set device for this process
    args.device = f'cuda:{rank}'
    torch.cuda.set_device(args.device)
    
    # Initialize environment and agents for this GPU
    agents_per_gpu = args.num_agents // world_size
    start_idx = rank * agents_per_gpu
    agents = [Agent(id=i, args=args) for i in range(start_idx, start_idx + agents_per_gpu)]
    env = MultiGridEnv(args, agents)
    
    # Initialize network and wrap with DDP
    network = GR_QNetwork(args)
    network.qnetwork_local = DDP(
        network.qnetwork_local.to(args.device), 
        device_ids=[rank],
        output_device=rank
    )
    network.qnetwork_target = network.qnetwork_target.to(args.device)
    
    if args.logger:
        logger = Logger(args) if rank == 0 else None
    else:
        logger = None
    
    # Run training
    try:
        _ = train(env, agents, network, logger, args)
    finally:
        cleanup()
        if logger: 
            logger.close()

def train(env, agents, network, logger, args):
    progress = 0
    step_count = 0
    episode_count = 0
    eps = args.eps_start
    batch_size = len(agents)
    dqn_losses = deque(maxlen=100)
    total_scores = deque(maxlen=100)
    steps_per_episode = deque(maxlen=100)
    seen_percentages = deque(maxlen=100)
    agent_ids = torch.tensor([agent.id for agent in agents], device=args.device)
    
    # Get process rank for distributed training
    rank = dist.get_rank() if args.distributed else 0

    while step_count < args.total_steps:
        episode_start_time = time.time()
        episode_count += 1
        scores = [0 for _ in range(len(agents))]
        
        # Synchronize GPUs at the start of each episode
        if args.distributed:
            dist.barrier()
        
        # Get initial observations in batched form
        obs_batch, node_obs_batch, adj_batch = env.reset(args.render)

        # Update network bias (PER)
        if step_count % 100 == 0:
            network.update_beta(step_count, args.total_steps, args.prio_b)

        for step in range(args.episode_steps):
            step_count += 1

            # Get actions for all agents in one forward pass
            actions = network.action(obs_batch, node_obs_batch, adj_batch, agent_ids, eps, args.debug)

            # Step environment with batched actions
            next_obs_batch, next_node_obs_batch, next_adj_batch, rewards, dones, info = env.step(actions, args.render)

            if args.debug:
                for i in range(batch_size):
                    print_info(agent_ids[i], obs_batch[i], node_obs_batch[i], adj_batch[i], 
                             actions[i], rewards[i], next_obs_batch[i], next_node_obs_batch[i], 
                             next_adj_batch[i], dones[i])

            # Synchronize before processing experiences
            if args.distributed:
                dist.barrier()

            # Process all experiences in one batch
            loss = network.step(agent_ids, obs_batch, node_obs_batch, adj_batch, 
                              actions, rewards, next_obs_batch, next_node_obs_batch, 
                              next_adj_batch, dones)
            
            if loss is not None and step_count > 10000:
                dqn_losses.append(loss.cpu().item())
            
            # Update scores
            for i in range(batch_size):
                scores[i] += rewards[i]

            # Update observations
            obs_batch = next_obs_batch
            node_obs_batch = next_node_obs_batch
            adj_batch = next_adj_batch

            # Update epsilon
            if eps > args.eps_end:
                eps -= args.eps_decay
            else:
                eps = args.eps_end

            if any(dones):
                seen_percentages.append(info["seen_percentage"])
                break

        # Log data only on rank 0 if distributed
        if rank == 0:
            episode_time = time.time() - episode_start_time
            batch_scores = torch.sum(torch.tensor(scores, device=args.device, dtype=torch.float32))
            total_scores.append(batch_scores)
            steps_per_episode.append(step)
            
            if logger:
                # Convert tensors to torch and calculate means on GPU
                avg_rewards = torch.mean(torch.tensor(total_scores, device=args.device, dtype=torch.float32)).item()
                avg_steps = torch.mean(torch.tensor(steps_per_episode, device=args.device, dtype=torch.float32)).item()
                avg_loss = torch.mean(torch.tensor(dqn_losses, device=args.device, dtype=torch.float32)).item() if dqn_losses else 0.0
                avg_seen = torch.mean(torch.tensor(seen_percentages, device=args.device, dtype=torch.float32)).item()
                
                logger.log_metrics(
                    total_steps=step_count,
                    episodes=episode_count,
                    epsilon=eps,
                    average_rewards=avg_rewards,
                    average_steps_per_episode=avg_steps,
                    average_loss=avg_loss,
                    goals_collected=info.get("goals_collected", 0),
                    goals_percentage=info.get("goals_percentage", 0),
                    seen_percentage=avg_seen,
                    episode_duration=episode_time
                )
                logger.log_agent_metrics(episode_count, scores)
            
            # Progress tracking and model saving
            current_progress = round((step_count / args.total_steps) * 100)
            
            # Calculate averages once and reuse
            avg_score = torch.mean(torch.tensor(total_scores, device=args.device, dtype=torch.float32)).item()
            avg_seen = torch.mean(torch.tensor(seen_percentages, device=args.device, dtype=torch.float32)).item()
            
            print(f'\r{current_progress}% | Eps: {eps:.2f} \tAvg Score: {avg_score:.2f} \tAvg Seen: {avg_seen:.2f}%', end="")
            if current_progress != progress:
                print(f'\r{current_progress}% | Eps: {eps:.2f} \tAvg Score: {avg_score:.2f} \tAvg Seen: {avg_seen:.2f}%')
                progress = current_progress
                torch.save(network.qnetwork_local.module.state_dict(), f'{args.title}.pt')

    # Save the learned policy
    if rank == 0:
        torch.save(network.qnetwork_local.module.state_dict(), f'{args.title}.pt')
    return scores

def main():
    # Settings & Controls
    args = Args()
    
    if args.distributed and torch.cuda.is_available():
        # Set up environment variables for distributed training
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Launch distributed processes
        torch.multiprocessing.set_start_method('spawn', force=True)
        mp.spawn(
            train_ddp,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    else:
        # Original single-GPU code
        agents = [Agent(id=i, args=args) for i in range(args.num_agents)]
        env = MultiGridEnv(args, agents)
        network = GR_QNetwork(args)
        
        if args.logger:
            logger = Logger(args)
        else:
            logger = None
        
        if not args.load_policy:
            _ = train(env, agents, network, logger, args)
            
        if args.logger: 
            logger.close()
        env.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted")
        cleanup()