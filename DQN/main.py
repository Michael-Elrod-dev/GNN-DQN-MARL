import torch
# import cProfile
import numpy as np

from args import Args
from logger import Logger
from collections import deque
from network import Network
from minigrid.world import Agent
from minigrid.environment import MultiGridEnv
from utils import print_info


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
        obs = env.reset(args.render)
        
        for step in range(args.episode_steps):
            total_steps += 1
            actions = []
            
            for i in range(len(agents)):
                action = network.action(obs[i].unsqueeze(0), 0.0, args.debug)
                actions.append(action)
            
            next_obs, step_rewards, dones = env.step(actions, args.render)
            
            for i in range(len(agents)):
                rewards[i] += step_rewards[i]
            
            obs = next_obs
            
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

    while step_count < args.total_steps:
        episode_count += 1
        scores = [0 for _ in range(len(agents))]
        obs = env.reset(args.render)

        # Update network bias (PER)
        if step_count % 100 == 0:
            network.update_beta(step_count, args.total_steps, args.prio_b)

        for step in range(args.episode_steps):
            actions = []
            step_count += 1

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
    
    # # Start profiling
    # profiler = cProfile.Profile()
    # profiler.enable()

    # Start Run
    if not args.load_policy:
        _ = train(env, agents, network, logger, args)
    _ = test(env, agents, network, args)

    # # Stop profiling
    # profiler.disable()

    # # Print profiling stats
    # profiler.print_stats(sort='time')

    # Close Environment
    if args.logger: logger.close()
    env.close()

if __name__ == '__main__':
    main()
