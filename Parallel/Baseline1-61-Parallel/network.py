import torch
import random
import numpy as np
import torch.optim as optim

from gnn import GNNBase
from utils import unbind, print_info
from collections import deque, namedtuple


class GR_QNetwork():
    def __init__(self, args):
        self.t_step = 0
        self.lr = args.lr
        self.tau = args.tau
        self.seed = args.seed
        self.gamma = args.gamma
        if args.priority_replay:
            self.prio_e = args.prio_e
            self.prio_a = args.prio_a
            self.prio_b = args.prio_b
        self.obs_size = args.obs_size
        self.num_agents = args.num_agents
        self.double_dqn = args.double_dqn
        self.batch_size = args.batch_size
        self.action_size = args.action_size
        self.update_step = args.update_step
        self.buffer_size = args.buffer_size
        self.device = torch.device(args.device)

        # Q-Network + Buffer
        self.qnetwork_local = GNNBase(args).to(self.device)
        self.qnetwork_target = GNNBase(args).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=args.lr)
        self.memory = ReplayBuffer(args)   

    @property
    def time_step(self):
        return self.t_step // self.num_agents
    
    def step(self, agent_id, obs, node_obs, adj, action, reward, next_obs, next_node_obs, next_adj, done):
        loss = None
        self.t_step += 1
        # Add experience to replay buffer
        self.memory.add(torch.tensor(agent_id), obs.float(), node_obs.float(), adj.float(), torch.tensor(action), torch.tensor(reward), next_obs.float(), next_node_obs.float(), next_adj.float(), torch.tensor(done))
        
        # Update the network
        if self.time_step % self.update_step == 0:
            if len(self.memory) > self.batch_size:
                experiences, experience_indexes, priorities = self.memory.sample()
                loss = self.learn(experiences, experience_indexes, priorities, self.gamma)
        return loss
   
    def action(self, obs, node_obs, adj, agent_id, eps, testing):
        obs = obs.float().to(self.device)
        node_obs = node_obs.float().to(self.device)
        adj = adj.float().to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(obs, node_obs, adj, agent_id)
        self.qnetwork_local.train()

        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            if testing:
                while True:
                    user_input = input("Enter an action (W,A,S,D): ")
                    if user_input == 'a':
                        action = 0  # Left
                        break
                    elif user_input == 'd':
                        action = 1  # Right
                        break
                    elif user_input == 'w':
                        action = 2  # Up
                        break
                    elif user_input == 's':
                        action = 3  # Down
                        break
                    else:
                        print("Invalid input. Please enter (W,A,S,D)")
            else:
                action = random.choice(range(self.action_size))
        return action

    def learn(self, experiences, experience_indexes, priorities, gamma):
        agent_id, obs, node_obs, adj, actions, rewards, next_obs, next_node_obs, next_adj, done = experiences
        
        # Calculate current Q(s,a)
        Q_s = self.qnetwork_local(obs, node_obs, adj, agent_id)
        Q_s_a = Q_s.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next obs) from target model
        Q_s_next = self.qnetwork_target(next_obs, next_node_obs, next_adj, agent_id).max(1)[0].unsqueeze(1)
        targets = rewards + gamma * Q_s_next * (1 - done.float())
        
        # Calculate loss between local and target network
        losses = (Q_s_a - targets)**2
        
        # Importance-sampling weights (Prioritized Experience Replay)
        importance_weights = (((1/self.buffer_size)*(1/priorities))**self.prio_b).unsqueeze(1)
        
        loss = (importance_weights*losses).mean()
        dqn_loss = loss

        # Calculate gradients and do a step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate priorities and update them
        target_priorities = abs(Q_s_a - targets).detach().cpu().numpy() + self.prio_e
        self.memory.update_priority(experience_indexes, target_priorities)

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return dqn_loss                 

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def update_beta(self, current_step, total_steps, beta_start):
        beta = beta_start + (1.0 - beta_start) * (current_step / total_steps)
        self.prio_b = min(beta, 1.0)


class ReplayBuffer:
    def __init__(self, args):
        self.action_size = args.action_size
        self.device = torch.device(args.device)
        self.memory = deque(maxlen=args.buffer_size)  
        self.priority = deque(maxlen=args.buffer_size)  
        self.batch_size = args.batch_size
        self.experience = namedtuple("Experience", field_names=["id", "obs", "node_obs", "adj", "action", "reward", "next_obs", "next_node_obs", "next_adj", "done"])
        self.seed = args.seed
        if args.priority_replay:
            self.prio_e = args.prio_e
            self.prio_a = args.prio_a
            self.prio_b = args.prio_b
    
    def add(self, id, obs, node_obs, adj, action, reward, next_obs, next_node_obs, next_adj, dones):
        e = self.experience(id, obs, node_obs, adj, action, reward, next_obs, next_node_obs, next_adj, dones)
        self.memory.append(e)
        self.priority.append(self.prio_e)
        
    def update_priority(self, priority_indexes, priority_targets):
        for index,priority_index in enumerate(priority_indexes):
            self.priority[priority_index] = priority_targets[index][0]
    
    def sample(self):
        adjusted_priority = torch.tensor(self.priority, dtype=torch.float32, device=self.device) ** self.prio_a
        sampling_probability = adjusted_priority / adjusted_priority.sum()
        experience_indexes = torch.multinomial(sampling_probability, self.batch_size, replacement=False)
        experiences = [self.memory[index] for index in experience_indexes]

        id = torch.stack([e.id for e in experiences if e is not None]).to(self.device)
        obs = torch.stack([e.obs for e in experiences if e is not None]).to(self.device)
        node_obs = torch.stack([e.node_obs for e in experiences if e is not None]).to(self.device)
        adj = torch.stack([e.adj for e in experiences if e is not None]).to(self.device)
        action = torch.stack([e.action for e in experiences if e is not None]).to(self.device)
        reward = torch.stack([e.reward for e in experiences if e is not None]).to(self.device)
        next_obs = torch.stack([e.next_obs for e in experiences if e is not None]).to(self.device)
        next_node_obs = torch.stack([e.next_node_obs for e in experiences if e is not None]).to(self.device)
        next_adj  = torch.stack([e.next_adj for e in experiences if e is not None]).to(self.device)
        done = torch.stack([e.done for e in experiences if e is not None]).to(self.device)
        priorities = torch.tensor([self.priority[index] for index in experience_indexes], dtype=torch.float32, device=self.device)

        return (id, obs, node_obs, adj, action, reward, next_obs, next_node_obs, next_adj, done), experience_indexes, priorities

    def __len__(self):
        return len(self.memory)
    