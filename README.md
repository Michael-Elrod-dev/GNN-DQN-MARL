# Multi-Agent Reinforcement Learning with Graph Neural Networks
## Report
- You can view the latest version of the paper on Overleaf [here](https://www.overleaf.com/7797839727krzmnzncwwzn#cd413e).
- You can view the project presentation I did at the MIT LL [here](https://github.com/Michael-Elrod-dev/GNN-DQN-MARL/blob/main/Presentation.pdf).


## Project Overview

This project implements a multi-agent reinforcement learning environment using graph neural networks (GNNs) for agent communication and coordination. The agents navigate in a grid-based environment, aiming to visit goal locations. The project includes multiple implementations with varying reward functions and architectures, as well as a baseline DQN implementation for comparison.

## Repository Structure

The repository is organized as follows:

```
├── DQN/
│   ├── DQN-4-20-21-150.py
│   └── ...etc.
├── GNN/
│   ├── GNN-4-20-21-150.py
│   └── ...etc.
└── README.md
```

Each directory contains a complete implementation of either the GNN or DQn implementation for a multi-agent system with variations in environments for testing purposes. The numbers following each file name represents the environments details:
DQN-4-20-21-150 = DQN network with 4 agents, 20 goals, 21x21 grid, 150 time steps per epoch

## Key Components

- **main.py**: Entry point of the program, handles training and evaluation.
- **network.py**: Implements the Q-network architecture.
- **gnn.py**: Defines the graph neural network layers and modules.
- **environment.py**: Defines the multi-agent grid environment.
- **args.py**: Configuration settings for the environment and training.
- **utils.py**: Utility functions and constants used throughout the project.
