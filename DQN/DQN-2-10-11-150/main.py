import time
import torch
import numpy as np
from GPUtil import GPUtil
from args import Args
from logger import Logger
from collections import deque
from network import Network
from minigrid.world import Agent
from minigrid.environment import MultiGridEnv
from utils import print_info

def monitor_gpu(step_count=None):
    # PyTorch memory stats (actively used memory)
    torch_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    torch_peak = torch.cuda.max_memory_allocated() / 1024**2   # MB
    torch_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
    
    # Overall GPU stats (from GPUtil)
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assuming first GPU
        gpu_total = gpu.memoryTotal
        gpu_used = gpu.memoryUsed
        gpu_free = gpu.memoryFree
        gpu_util = gpu.load * 100
    
    # Compile stats
    stats = {
        'torch_allocated_mb': torch_allocated,
        'torch_peak_mb': torch_peak,
        'torch_reserved_mb': torch_reserved,
        'gpu_total_mb': gpu_total,
        'gpu_used_mb': gpu_used,
        'gpu_free_mb': gpu_free,
        'gpu_utilization': gpu_util
    }
    
    step_info = f"Step {step_count} | " if step_count is not None else ""
    print(f"\r{step_info}"
            f"PyTorch Active: {torch_allocated:.0f}MB | "
            f"Peak: {torch_peak:.0f}MB | "
            f"Reserved: {torch_reserved:.0f}MB | "
            f"GPU Used: {gpu_used:.0f}/{gpu_total:.0f}MB | "
            f"Util: {gpu_util:.1f}%", end="")
    
    return stats

def train(env, agents, network, logger, args):
    training_start_time = time.time()
    
    progress = 0
    step_count = 0
    episode_count = 0
    eps = args.eps_start
    batch_size = len(agents)
    dqn_losses = deque(maxlen=100)
    total_scores = deque(maxlen=100)
    steps_per_episode = deque(maxlen=100)
    seen_percentages = deque(maxlen=100)
    agent_id = [agent.id for agent in agents]

    while step_count < args.total_steps:
        episode_start_time = time.time()
        episode_count += 1
        scores = [0 for _ in range(len(agents))]
        obs = env.reset(args.render)

        # Update network bias (PER)
        if step_count % 100 == 0:
            network.update_beta(step_count, args.total_steps, args.prio_b)

        for step in range(args.episode_steps):
            actions = []
            step_count += 1

            # Memory monitoring at each step
            # monitor_gpu(step_count)

            # Get actions for each agent
            for i in range(batch_size):
                actions.append(network.action(obs[i].unsqueeze(0), eps, args.debug))

            # Act in the environment
            next_obs, rewards, dones, info = env.step(actions, args.render)

            if args.debug:
                for i in range(batch_size):
                    print_info(agent_id[i], obs[i], actions[i], rewards[i], next_obs[i], dones[i])

            # Send experience to network
            for i in range(batch_size):
                loss = network.step(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
                if loss is not None and step_count > 10000:
                    dqn_losses.append(loss.cpu().item())
                scores[i] += rewards[i]

            obs = next_obs

            # Update epsilon
            if eps > args.eps_end:
                eps -= args.eps_decay
            else:
                eps = args.eps_end

            if any(dones):
                seen_percentages.append(info["seen_percentage"])
                break

        episode_time = time.time() - episode_start_time
        total_training_time = time.time() - training_start_time

        # Log data
        total_scores.append(sum(scores))
        steps_per_episode.append(step)
        if logger:
            logger.log_metrics(
                total_steps=step_count,
                episodes=episode_count,
                epsilon=eps,
                average_rewards=np.mean(total_scores),
                average_steps_per_episode=np.mean(steps_per_episode),
                average_loss=np.mean(dqn_losses) if dqn_losses else 0.0,
                goals_collected=info.get("goals_collected", 0),
                goals_percentage=info.get("goals_percentage", 0),
                seen_percentage=np.mean(seen_percentages),
                episode_duration=episode_time,
                episode_duration_minutes=episode_time / 60,
                total_training_time=total_training_time,
                total_training_time_minutes=total_training_time / 60,
                total_training_time_hours=total_training_time / 3600,
                steps_per_second=step_count / total_training_time
            )

        current_progress = round((step_count / args.total_steps) * 100)
        print(f'\r{current_progress}% | Eps: {eps:.2f} \tAvg Score: {np.mean(total_scores):.2f} \tAvg Seen: {np.mean(seen_percentages):.2f}%', end="")
        if current_progress != progress:
            print(f'\r{current_progress}% | Eps: {eps:.2f} \tAvg Score: {np.mean(total_scores):.2f} \tAvg Seen: {np.mean(seen_percentages):.2f}%')
            progress = current_progress
            torch.save(network.qnetwork_local.state_dict(), f'{args.title}.pt')

    # Save the learned policy
    torch.save(network.qnetwork_local.state_dict(), f'{args.title}.pt')
    return scores

def main():
    # Settings & Controls
    args = Args()
    agents = [Agent(id=i, args=args) for i in range(args.num_agents)]
    env = MultiGridEnv(args, agents)
    network = Network(args)
    
    if args.logger:
        logger = Logger(args)
    else:
        logger = None

    # Start Run
    if not args.load_policy:
        _ = train(env, agents, network, logger, args)

    # Close Environment
    if args.logger: logger.close()
    env.close()

if __name__ == '__main__':
    main()