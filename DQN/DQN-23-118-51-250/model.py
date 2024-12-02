import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, args):
        super(QNetwork, self).__init__()
        # input
        self.seed = torch.manual_seed(args.seed)
        self.fc1 = nn.Linear(args.obs_size, 64)
        # advantage
        self.ad1 = nn.Linear(64, 32)
        self.ad2 = nn.Linear(32, args.action_size)
        # value
        self.va1 = nn.Linear(64, 32)
        self.va2 = nn.Linear(32, 1)

    def forward(self, obs):
        # input
        linear_1 = F.relu(self.fc1(obs))
        # advantage
        advantage_1 = F.relu(self.ad1(linear_1))
        action_advantage = self.ad2(advantage_1)
        # value
        value_1 = F.relu(self.va1(linear_1))
        obs_value = self.va2(value_1)
        # combining
        max_action_advantage = torch.max(action_advantage, dim=1)[0].unsqueeze(1)
        value_obs_action = obs_value + action_advantage - max_action_advantage 
        
        return value_obs_action
    
    def count_layers_and_parameters(self):
        print("Layer Summary:")
        total_params = 0
        for name, module in self.named_children():
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name}: {module.__class__.__name__}, Parameters: {num_params}")
            total_params += num_params

        print(f"\nTotal Summary:")
        print(f"  Number of layers: {len(list(self.children()))}")
        print(f"  Total trainable parameters: {total_params}")

        return len(list(self.children())), total_params