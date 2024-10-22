import torch
import cProfile
import pstats
import io
import numpy as np

from args import Args
from logger import Logger
from collections import deque
from network import GR_QNetwork
from minigrid.world import Agent
from minigrid.environment import MultiGridEnv
from utils import print_info

assert torch.cuda.is_available(), "CUDA is not available. Please check your installation."



def test(env, agents, network, args):
    network.qnetwork_local.load_state_dict(torch.load(f'policy/{args.title}.pt'))
    print(f'-' * 40)
    print('Model Evaluation')
    episode_count = 0
    total_steps = 0
    total_rewards = []
    steps_per_episode = []
    
    while episode_count < 100:
        episode_count += 1
        rewards = [0 for _ in range(len(agents))]
        obs, node_obs, adj = env.reset(args.render)
        
        for step in range(args.episode_steps):
            total_steps += 1
            actions = []
            
            for i in range(len(agents)):
                action = network.action(obs[i].unsqueeze(0), node_obs[i].unsqueeze(0), adj[i].unsqueeze(0), agents[i].id, 0.0, args.debug)
                actions.append(action)
            
            next_obs, next_node_obs, next_adj, step_rewards, dones = env.step(actions, args.render)
            
            for i in range(len(agents)):
                rewards[i] += step_rewards[i]
            
            obs = next_obs
            node_obs = next_node_obs
            adj = next_adj
            
            if any(dones):
                break
        
        total_rewards.append(sum(rewards))
        steps_per_episode.append(step)
        print(f"Episode {episode_count}: Total Reward = {sum(rewards)}, Steps = {step}")
    
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(steps_per_episode)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps per Episode: {avg_steps:.2f}")

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
    agent_id = [agent.id for agent in agents]

    while step_count < 600:  # Adjusted to your shorter training process
        episode_count += 1
        scores = [0 for _ in range(len(agents))]
        obs, node_obs, adj = env.reset(args.render)

        # Update network bias (PER)
        if step_count % 100 == 0:
            network.update_beta(step_count, 600, args.prio_b)  # Adjusted total steps

        for step in range(args.episode_steps):
            actions = []
            step_count += 1

            # Get actions for each agent
            for i in range(batch_size):
                actions.append(network.action(obs[i].unsqueeze(0), node_obs[i].unsqueeze(0), adj[i].unsqueeze(0), agent_id[i], eps, args.debug))

            # Act in the environment
            next_obs, next_node_obs, next_adj, rewards, dones, info = env.step(actions, args.render)

            if args.debug:
                for i in range(batch_size):
                    print_info(agent_id[i], obs[i], node_obs[i], adj[i], actions[i], rewards[i], next_obs[i], next_node_obs[i], next_adj[i], dones[i])

            # Send experience to network
            for i in range(batch_size):
                loss = network.step(agent_id[i], obs[i], node_obs[i], adj[i], actions[i], rewards[i], next_obs[i], next_node_obs[i], next_adj[i], dones[i])
                if loss is not None:
                    dqn_losses.append(loss.cpu().item())
                scores[i] += rewards[i]

            obs = next_obs
            node_obs = next_node_obs
            adj = next_adj

            # Update epsilon
            if eps > args.eps_end:
                eps -= args.eps_decay
            else:
                eps = args.eps_end

            # Print GPU usage every 100 steps
            if step_count % 100 == 0:
                print(f"\nStep {step_count}:")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

            if any(dones):
                seen_percentages.append(info["seen_percentage"])
                break

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
                seen_percentage=np.mean(seen_percentages)
            )
            logger.log_agent_metrics(episode_count, scores)

        current_progress = round((step_count / 600) * 100)  # Adjusted for 600 steps
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
    network = GR_QNetwork(args)
    
    if args.logger:
        logger = Logger(args)
    else:
        logger = None
    
    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Start Run
    if not args.load_policy:
        _ = train(env, agents, network, logger, args)
    # _ = test(env, agents, network, args)

    # Stop profiling
    profiler.disable()

    # Save profiling results to a file
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    with open('profiler_results.txt', 'w') as f:
        f.write(s.getvalue())

    # Close Environment
    if args.logger: logger.close()
    env.close()

if __name__ == '__main__':
    main()
