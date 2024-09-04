from __future__ import annotations

import time
import math
import copy
import torch
import pygame
import numpy as np
import gymnasium as gym

from abc import abstractmethod
from minigrid.grid import Grid
from typing import Iterable, TypeVar
from minigrid.world import Goal, Agent, Obstacle
from utils import Actions, COLOR_NAMES, TILE_PIXELS


T = TypeVar("T")

class MultiGridEnv(gym.Env):
    def __init__(self, args, agents):
        self.clock = None
        self.window = None
        self.agents = agents
        self.goals = []
        self.obstacles = []
        self.seed = args.seed
        self.actions = Actions
        self.screen_size = 640
        self.render_size = None
        self.gamma = args.gamma

        # Environment configuration
        self.entities = []
        self.step_count = 0
        self.num_collected = 0
        self.distance_matrix = []
        self.width = args.grid_size
        self.height = args.grid_size
        self.see_through_walls = True
        self.num_goals = args.num_goals
        self.max_steps = args.episode_steps
        self.num_obstacles = args.num_obstacles
        self.max_edge_dist = args.max_edge_dist
        self.full_features = args.full_features
        self.grid = Grid(self.width, self.height)
        self.agent_view_size = args.agent_view_size
        
        # Rendering attributes
        self.highlight = True
        self.tile_size = TILE_PIXELS
        self.window_name = "Custom MiniGrid"

        # Dummy Class
        self.dummy_goal = Goal()
        self.dummy_goal.cur_pos = (0, 0)
        self.dummy_goal.collected = False
        
    def reset(self, render):
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        super().reset(seed=int(time.time()))
        adj = []
        obs = []
        node_obs = []
        self.goals = []
        self.entities = []
        self.obstacles = []
        self.dummy_goal = Goal()
        self.dummy_goal.cur_pos = (0, 0)
        self.dummy_goal.collected = False
        
        self.step_count = 0
        self.num_collected = 0

        for agent in self.agents:
            agent.reset()

        self._gen_grid(self.width, self.height)
        self.calc_distances(True)

        for agent in self.agents:
            obs.append(self._get_obs(agent))
        for agent in self.agents:
            node_obs.append(self._get_node_features(agent))
        for agent in self.agents:
            adj.append(self.distance_matrix)

        if render: 
            self.render()
        
        return obs, node_obs, adj

    @abstractmethod
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place agents at the bottom of the grid with maximum space between them
        for index, agent in enumerate(self.agents):
            if len(self.agents) > 1:
                interval = (width - 3) // (len(self.agents) - 1)
                x = 1 + interval * index
            else:
                x = (width - 1) // 2
            y = height - 2
            agent.init_pos = (x, y)
            agent.cur_pos = (x, y)
            agent.direction = 3
            self.grid.set_agent(x, y, agent)
            self.entities.append(agent)

        # Place goals randomly in the grid
        for _ in range(self.num_goals):
            obj = Goal()
            while True:
                x, y = self._rand_pos(0, width - 1, 0, height - 2)
                if not self.grid.get(x, y):
                    self.grid.set(x, y, obj)
                    obj.cur_pos = (x, y)
                    self.entities.append(obj)
                    self.goals.append(obj)
                    break

        # Place obstacles randomly in the grid
        for _ in range(self.num_obstacles):
            obj = Obstacle()
            while True:
                x, y = self._rand_pos(0, width - 1, 0, height - 2)
                if not self.grid.get(x, y):
                    self.grid.set(x, y, obj)
                    obj.cur_pos = (x, y)
                    self.entities.append(obj)
                    self.obstacles.append(obj)
                    break
  
    def _get_obs(self, agent):
        obs = []
        obs.extend(agent.cur_pos)

        agent_id = agent.id
        num_agents = len(self.agents)
        num_goals = len(self.goals)

        # Find goals within the agent's FOV and store their distances
        goals_in_fov = []
        for goal_id in range(num_goals):
            goal = self.goals[goal_id]
            dist = self.distance_matrix[agent_id, num_agents + goal_id]
            collected = self.goals[goal_id].collected

            if dist <= self.max_edge_dist and not collected:
                goals_in_fov.append((goal, dist))

        # Sort goals by distance in ascending order
        goals_in_fov.sort(key=lambda x: x[1])
        goals_in_fov = goals_in_fov[:3]
        
        # Fill the remaining positions with dummy goals if necessary
        if len(goals_in_fov) == 0:
            goals_in_fov = [(self.dummy_goal, float('inf'))] * 3
        elif len(goals_in_fov) == 1:
            goals_in_fov.extend([(self.dummy_goal, float('inf'))] * 2)
        elif len(goals_in_fov) == 2:
            goals_in_fov.append((self.dummy_goal, float('inf')))

        # Check if any of the agent's goals have been collected
        if agent.goal1 is not None and agent.goal1.collected:
            agent.goal1 = self.dummy_goal
        if agent.goal2 is not None and agent.goal2.collected:
            agent.goal2 = self.dummy_goal
        if agent.goal3 is not None and agent.goal3.collected:
            agent.goal3 = self.dummy_goal

        # Scenario 1: Start of a new episode (all agent.goal locations are None)
        if agent.goal1 is None and agent.goal2 is None and agent.goal3 is None:
            if len(goals_in_fov) > 0:
                agent.goal1 = goals_in_fov[0][0]
            if len(goals_in_fov) > 1:
                agent.goal2 = goals_in_fov[1][0]
            if len(goals_in_fov) > 2:
                agent.goal3 = goals_in_fov[2][0]

        # Scenario 2: Step within an epoch (update goal locations based on distances)
        elif agent.goal1 is not None and agent.goal2 is not None and agent.goal3 is not None:
            for goal, dist in goals_in_fov:
                if goal == agent.goal1 or goal == agent.goal2 or goal == agent.goal3:
                    continue
                if agent.goal1 == self.dummy_goal or dist < self.distance_matrix[agent_id, num_agents + self.goals.index(agent.goal1)]:
                    agent.goal3 = agent.goal2
                    agent.goal2 = agent.goal1
                    agent.goal1 = goal
                elif agent.goal2 == self.dummy_goal or dist < self.distance_matrix[agent_id, num_agents + self.goals.index(agent.goal2)]:
                    agent.goal3 = agent.goal2
                    agent.goal2 = goal
                elif agent.goal3 == self.dummy_goal or dist < self.distance_matrix[agent_id, num_agents + self.goals.index(agent.goal3)]:
                    agent.goal3 = goal
        else:
            # Fill remaining goal variables with dummy goal objects
            if agent.goal1 is None:
                agent.goal1 = self.dummy_goal
            if agent.goal2 is None:
                agent.goal2 = self.dummy_goal
            if agent.goal3 is None:
                agent.goal3 = self.dummy_goal

        # Add goal information to the observation vector
        obs.extend(agent.goal1.cur_pos)
        # obs.append(int(agent.goal1.collected))
        obs.extend(agent.goal2.cur_pos)
        # obs.append(int(agent.goal2.collected))
        obs.extend(agent.goal3.cur_pos)
        # obs.append(int(agent.goal3.collected))

        agent.obs = torch.tensor(obs)
        return agent.obs
    
    # TODO: fix numpy to tensor conversion
    def _get_node_features(self, agent):
        features = []
        agent_pos = agent.cur_pos

        for entity in self.entities:
            entity_features = []
            entity_pos = entity.cur_pos
            rel_pos = [entity_pos[0] - agent_pos[0], entity_pos[1] - agent_pos[1]]
            entity_features.extend(rel_pos)
            if entity.type == 'agent':
                # Get the goal locations and statuses from the other agent
                other_agent_goals = [entity.goal1, entity.goal2, entity.goal3]
                for goal in other_agent_goals:
                    rel_goal_pos = [goal.cur_pos[0] - agent_pos[0], goal.cur_pos[1] - agent_pos[1]]
                    entity_features.extend(rel_goal_pos)
                    if self.full_features:
                        entity_features.append(int(goal.collected))
            else:
                for _ in range(3):
                    entity_features.extend(rel_pos)
                    if self.full_features:
                        entity_features.append(0)

            entity_features.append(0 if entity.type == 'agent' else (1 if entity.type == 'goal' else 2))
            features.append(entity_features)
        # [[[rel_pos, rel_goal_pos1, rel_goal_pos1_completed, ..., entity_type], [rel_pos, rel_goal_pos2, rel_goal_pos2_completed, ...], ...], ...]
        return torch.tensor(np.array(features))

    def _handle_overlap(self, i, fwd_pos, fwd_cell):
        reward = 0
        if isinstance(fwd_cell, Goal) and not fwd_cell.collected:
            reward = self._reward(5)
            fwd_cell.color = 'grey'
            fwd_cell.collected = True
            self.num_collected += 1
        elif isinstance(fwd_cell, Goal) and fwd_cell.collected:
            pass
            # reward = self._reward(-5)
        elif isinstance(fwd_cell, Obstacle):
            reward = self._reward(-1)
        else:
            print(f'fwd_cell = {fwd_cell}')
            raise ValueError("_handle_overlap error.")

        self.grid.set_agent(*fwd_pos, self.agents[i])
        self.grid.set_agent(*self.agents[i].cur_pos, None)
        self.agents[i].cur_pos = fwd_pos
        return reward

    def step(self, actions, render):
        adj = []
        obs = []
        node_obs = []
        dones = [False] * len(self.agents)
        self.step_count += 1
        rewards = np.zeros(len(self.agents))

        for i, action in enumerate(actions):
            agent_pos = self.agents[i].cur_pos
            
            # Move up
            if action == self.actions.up:
                new_pos = (agent_pos[0], agent_pos[1] - 1)
            # Move down
            elif action == self.actions.down:
                new_pos = (agent_pos[0], agent_pos[1] + 1)
            # Move left
            elif action == self.actions.left:
                new_pos = (agent_pos[0] - 1, agent_pos[1])
            # Move right
            elif action == self.actions.right:
                new_pos = (agent_pos[0] + 1, agent_pos[1])

            # Check if the new position is valid (not a wall and within bounds)
            cell = self.grid.get(*new_pos)
            agent = self.grid.get_agent(*new_pos)

            # If move is valid and space is empty
            if cell is None and agent is None and 0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height:
                self.grid.set_agent(*new_pos, self.agents[i])
                self.grid.set_agent(*self.agents[i].cur_pos, None)
                self.agents[i].cur_pos = new_pos
            # If move is valid but involves an overlap
            elif cell and cell.can_overlap():
                rewards[i] = self._handle_overlap(i, new_pos, cell)
            # If move is not valid
            else:
                rewards[i] = self._reward(-1)

            if self.step_count >= self.max_steps or self.num_collected >= self.num_goals:
                dones[i] = True

        self.calc_distances(False)
        dones = [True if any(dones) else d for d in dones]
    
        for agent in self.agents:
            obs.append(self._get_obs(agent))
        for agent in self.agents:
            node_obs.append(self._get_node_features(agent))
        for agent in self.agents:
            adj.append(self.distance_matrix)

        if render:
            self.render()
        
        return obs, node_obs, adj, rewards, dones
    
    def calc_distances(self, reset=False):
        num_entities = len(self.entities)
        num_agents = len(self.agents)
        
        if reset:
            # Calculate distances between all entities
            distance_matrix = torch.zeros((num_entities, num_entities))
            for i in range(num_entities):
                for j in range(i + 1, num_entities):
                    entity1 = self.entities[i]
                    entity2 = self.entities[j]
                    x1, y1 = entity1.cur_pos
                    x2, y2 = entity2.cur_pos
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
        else:
            # Update distances only between agents and other entities
            distance_matrix = self.distance_matrix.clone()
            for i in range(num_agents):
                for j in range(num_agents, num_entities):
                    agent = self.agents[i]
                    entity = self.entities[j]
                    x1, y1 = agent.cur_pos
                    x2, y2 = entity.cur_pos
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
        
        self.distance_matrix = distance_matrix
    
    def render(self):
        img = self.get_full_render(self.highlight, self.tile_size)
        img = np.transpose(img, axes=(1, 0, 2))

        if self.render_size is None:
            self.render_size = img.shape[:2]
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("minigrid")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(img)

        # Create background
        offset = surf.get_size()[0] * 0.1
        bg = pygame.Surface((int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset)))
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))
        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))
        line_surface = pygame.Surface((self.screen_size, self.screen_size), pygame.SRCALPHA)

        # Draw lines between agents without considering distance
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent1 = self.agents[i]
                agent2 = self.agents[j]
                pos1 = (((agent1.cur_pos[0] + 1) * self.tile_size + self.tile_size // 2) * (self.screen_size / (surf.get_size()[0] + offset)), 
                        ((agent1.cur_pos[1]) * self.tile_size + self.tile_size // 2) * (self.screen_size / (surf.get_size()[1] + offset)))
                pos2 = (((agent2.cur_pos[0] + 1) * self.tile_size + self.tile_size // 2) * (self.screen_size / (surf.get_size()[0] + offset)), 
                        ((agent2.cur_pos[1]) * self.tile_size + self.tile_size // 2) * (self.screen_size / (surf.get_size()[1] + offset)))
                pygame.draw.line(line_surface, (255, 0, 0), pos1, pos2, 2)

        # Draw lines between agents and non-agent entities within the specified distance
        for agent in self.agents:
            for entity in self.entities:
                if entity not in self.agents:
                    dist = math.sqrt((agent.cur_pos[0] - entity.cur_pos[0])**2 + (agent.cur_pos[1] - entity.cur_pos[1])**2)
                    if dist <= self.max_edge_dist:
                        pos1 = (((agent.cur_pos[0] + 1) * self.tile_size + self.tile_size // 2) * (self.screen_size / (surf.get_size()[0] + offset)), 
                                ((agent.cur_pos[1]) * self.tile_size + self.tile_size // 2) * (self.screen_size / (surf.get_size()[1] + offset)))
                        pos2 = (((entity.cur_pos[0] + 1) * self.tile_size + self.tile_size // 2) * (self.screen_size / (surf.get_size()[0] + offset)), 
                                ((entity.cur_pos[1]) * self.tile_size + self.tile_size // 2) * (self.screen_size / (surf.get_size()[1] + offset)))
                        pygame.draw.line(line_surface, (255, 0, 0), pos1, pos2, 2)

        # Blit the line surface onto the background
        bg.blit(line_surface, (0, 0))
        self.window.blit(bg, (0, 0))
        pygame.event.pump()
        pygame.display.flip()

    def get_full_render(self, highlight, tile_size):
        highlight_masks = np.zeros((self.width, self.height), dtype=bool)

        for i in range(self.width):
            for j in range(self.height):
                # Compute the Euclidean distance from the agent's position
                for agent in self.agents:
                    dist = math.sqrt((i - agent.cur_pos[0])**2 + (j - agent.cur_pos[1])**2)
                    if dist <= self.max_edge_dist:
                        highlight_masks[i, j] = True
        
        img = self.grid.render(tile_size,highlight_mask=highlight_masks if highlight else None)
        return img

    def _reward(self, reward):
        return reward * (self.gamma ** self.step_count)
    
    def _rand_int(self, low, high):
        return self.np_random.integers(low, high)

    def _rand_float(self, low, high):
        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        return (self.np_random.integers(0, 2) == 0)

    def _rand_elem(self, iterable: Iterable[T]):
        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int):
        lst = list(iterable)
        assert num_elems <= len(lst)
        out: list[T] = []
        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)
        return out

    def _rand_color(self):
        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        return (self.np_random.integers(xLow, xHigh),self.np_random.integers(yLow, yHigh))
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
