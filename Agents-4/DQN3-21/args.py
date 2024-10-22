import os
import time


class Args:
    def __init__(self):
        # Env Parameters
        self.env              = None
        self.num_agents       = 4
        self.num_goals        = 20
        self.num_obstacles    = 0
        self.grid_size        = 21
        self.agent_view_size  = 9
        self.action_size      = 4
        self.max_edge_dist    = 4.5
        self.obs_size         = 8

        # DQN Parameters
        self.prio_e           = 0.1
        self.prio_a           = 0.5
        self.prio_b           = 0.4
        self.double_dqn       = True
        self.priority_replay  = True

        # Training parameters
        self.total_steps      = 2500000
        self.episode_steps    = 150
        self.eps_start        = 1.0
        self.eps_end          = 0.01
        self.eps_percentage   = 0.80
        self.seed             = int(time.time())

        # Network Parameters
        self.buffer_size  = 100000   # Replay buffer size
        self.batch_size   = 64       # Sample batch size
        self.gamma        = 0.99     # Discount factor
        self.tau          = 0.001    # Soft update of target parameters
        self.lr           = 0.0005   # Learning rate 
        self.update_step  = 4        # Update the network

        # Run Parameters
        self.device       = 'cuda'
        self.load_policy  = False    # Evaluate a learned policy
        self.logger       = True     # Log training data to Wandb
        self.render       = False     # Render the environment
        self.debug        = False     # Allow manual action inputs

        # Reward Parameters
        self.reward_goal = 10
        self.penalty_goal = -1
        self.penalty_obstacle = -5
        self.penalty_invalid_move = -5

        # Derived Parameters
        self.title = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        self.eps_decay = self.calc_eps_decay(self.eps_start, self.eps_end, self.total_steps, self.eps_percentage)

       
    # Calculate the rate that ε should decay
    def calc_eps_decay(self, eps_start, eps_end, n_steps, eps_percentage):
        effective_steps = n_steps * eps_percentage
        decrement_per_step = (eps_start - eps_end) / effective_steps
        return decrement_per_step