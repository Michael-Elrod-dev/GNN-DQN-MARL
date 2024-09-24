import wandb
import socket


class Logger:
    def __init__(self, args):
        wandb.init(project='GNN-MARL-Test', entity='elrod-michael95', name=f'{args.title}', mode="online")
        
    def log_metrics(self, total_steps, episodes, epsilon, average_rewards, average_steps_per_episode, average_loss):
        wandb.log({
            "Episodes": episodes,
            "Total Steps": total_steps,
            "Epsilon": epsilon,
            "Average Reward": average_rewards,
            "Average Steps per Episode": average_steps_per_episode,
            "Average Loss": average_loss
        })

    def log_agent_metrics(self, episodes, agents_rewards):
        log_data = {"Episodes": episodes}
        for agent_id, reward in enumerate(agents_rewards):
            metric_name = f"Agent_{agent_id}/Reward"
            log_data[metric_name] = reward
        wandb.log(log_data)

    def log_test_metrics(self, episodes, epsilon, rewards, average_rewards, steps_per_episode, average_steps_per_episode):
        wandb.log({
            "Test/Steps per Episode": steps_per_episode,
            "Test/Average Steps per Episode": average_steps_per_episode,
            "Test/Episodes": episodes,
            "Test/Epsilon": epsilon,
            "Test/Reward": rewards,
            "Test/Average Reward": average_rewards
        })

    def check_connection(self):
        try:
            socket.create_connection(("www.google.com", 80))
            return True
        except OSError:
            return False

    def close(self):
        wandb.finish()
