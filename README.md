# Multi-Agent Reinforcement Learning with Graph Neural Networks
## Report
- You can view the latest version of the paper on Overleaf [here](https://www.overleaf.com/7797839727krzmnzncwwzn#cd413e).
- You can view the project presentation I did at the MIT LL [here](https://github.com/Michael-Elrod-dev/GNN-DQN-MARL/blob/main/Presentation.pdf).


## Project Overview

This project implements a multi-agent reinforcement learning environment using graph neural networks (GNNs) for agent communication and coordination. The agents navigate in a grid-based environment, aiming to collect goals while avoiding obstacles. The project includes multiple implementations with varying reward functions and architectures, as well as a baseline DQN implementation for comparison.

## Repository Structure

The repository is organized as follows:

```
├── Baseline/
│   ├── main.py
│   ├── network.py
│   ├── gnn.py
│   ├── environment.py
│   ├── args.py
│   └── utils.py
├── Baseline1/
│   └── [Different reward function]
├── Baseline2/
│   └── [Different reward function]
├── ...
├── DQN/
│   ├── main.py
│   ├── network.py
│   └── environment.py
├── Presentation.pdf
└── README.md
```

Each `Baseline` directory contains a complete implementation of the GNN-based multi-agent system with variations in reward functions or network architectures for testing purposes. The `DQN` directory contains a simpler implementation using standard Deep Q-Networks for comparison.

## Key Components

- **main.py**: Entry point of the program, handles training and evaluation.
- **network.py**: Implements the GNN-based Q-network architecture.
- **gnn.py**: Defines the graph neural network layers and modules.
- **environment.py**: Defines the multi-agent grid environment.
- **args.py**: Configuration settings for the environment and training.
- **utils.py**: Utility functions and constants used throughout the project.

## Installation

To run this project, you'll need Python 3.7+ and the following dependencies:
```bash
pip install torch torch-geometric numpy pygame gymnasium
```
## Usage
To train the model:
```bash
python main.py
```
To evaluate a trained model, set load_policy = True in args.py and run:
```bash
python main.py
```

## Experiments

The different Baseline directories allow for easy comparison of various reward functions and architectures. To run experiments with different setups:

- Navigate to the desired Baseline directory.
- Adjust parameters in `args.py` as needed.
- Run `main.py` to train and evaluate the model.
- Compare results across different baselines and with the DQN implementation.

##  Results
ToDo
