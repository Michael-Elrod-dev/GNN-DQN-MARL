import wandb
import socket

class Logger:
    def __init__(self, args):
        wandb.init(
            project='GNN-Test', 
            entity='elrod-michael95', 
            name=f'{args.title}', 
            mode="online",
            config={
                "num_agents": args.num_agents,
                "num_goals": args.num_goals,
                "grid_size": args.grid_size
            }
        )
        
    def log_metrics(self, **metrics):
        log_data = {
            "Episodes": metrics.get("episodes"),
            "Total Steps": metrics.get("total_steps"),
            "Epsilon": metrics.get("epsilon"),
            "Average Reward": metrics.get("average_rewards"),
            "Average Steps per Episode": metrics.get("average_steps_per_episode"),
            "Average Loss": metrics.get("average_loss"),
            "Goals Collected": metrics.get("goals_collected"),
            "Goals Collected Percentage": metrics.get("goals_percentage"),
            "Grid Seen Percentage": metrics.get("seen_percentage"),
            
            # Timing metrics
            "Episode Duration (s)": metrics.get("episode_duration"),
            "Episode Duration (min)": metrics.get("episode_duration_minutes"),
            "Total Training Time (s)": metrics.get("total_training_time"),
            "Total Training Time (min)": metrics.get("total_training_time_minutes"),
            "Total Training Time (h)": metrics.get("total_training_time_hours"),
            "Training Throughput (steps/s)": metrics.get("steps_per_second"),
        }
        
        # Remove None values
        log_data = {k: v for k, v in log_data.items() if v is not None}
        wandb.log(log_data)

    def check_connection(self):
        try:
            socket.create_connection(("www.google.com", 80))
            return True
        except OSError:
            return False

    def close(self):
        wandb.finish()