# Multi-Agent Reinforcement Learning with Graph Neural Networks

This project implements a multi-agent reinforcement learning environment using graph neural networks (GNNs) for agent communication and coordination. The agents navigate in a grid-based environment, aiming to collect goals while avoiding obstacles.

## Features

- Multi-agent environment with customizable grid size, number of agents, goals, and obstacles
- Graph neural network architecture for agent communication and decision-making
- Prioritized experience replay for efficient learning
- Visualization of the environment using Pygame

## Requirements

- Python 3.x
- PyTorch
- Numpy
- Pygame
- Gymnasium

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Michael-Elrod-dev/GNN-DQN-MARL.git
   ```

2. Install the required dependencies: ToDo
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Modify the configuration settings in `args.py` as needed.

2. Run the training script:
   ```
   python main.py
   ```

3. To evaluate a trained model, set `load_policy` to `True` in `args.py` and run:
   ```
   python main.py
   ```

## Code Structure

- `args.py`: Configuration settings for the environment and training.
- `main.py`: Entry point of the program, handles training and evaluation.
- `environment.py`: Defines the multi-agent grid environment.
- `network.py`: Implements the graph neural network architecture.
- `gnn.py`: Defines the graph neural network layers and modules.
- `logger.py`: Handles logging of training metrics using Weights and Biases (wandb).
- `utils.py`: Utility functions and constants used throughout the project.
- `grid.py`: Defines the grid structure and rendering.
- `rendering.py`: Utility functions for image rendering.
- `world.py`: Defines the world objects (agents, goals, obstacles, walls).
