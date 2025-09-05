# Donet Humanoid Control

A PyTorch implementation of Donet (Dynamic Oscillatory Neural Embodied Tuning) for humanoid robot walking in MuJoCo environments, based on generalized REMI architecture.

## Overview

This project implements the humanoid walking system described in CLAUDE.md, which adapts the REMI architecture into Donet for continuous robot control. The system uses:

- **Modified GRU Architecture**: Specialized unit groups for joint states, perceptual processing, place associations, planning, and torque control
- **Difference of Gaussians (DoG) Encoding**: Sparse encoding of joint angles with lateral inhibition
- **Curriculum Training**: 3-phase training from pure exploration to refined control
- **PPO RL Bridge**: High-level walking guidance via Proximal Policy Optimization
- **Intertwined Planning-Execution**: Continuous plan-act-replan cycles during operation

## Project Structure

```
donet_control/
├── config/                 # Hydra configuration files
│   ├── config.yaml        # Main configuration
│   ├── model/             # Model configurations
│   ├── task/              # Task configurations  
│   └── training/          # Training configurations
├── src/
│   ├── models/            # Neural network architectures
│   │   ├── encoding.py    # DoG and perceptual encoders
│   │   └── donet_gru.py   # Donet GRU implementation
│   ├── tasks/             # Task definitions
│   │   ├── base_task.py   # Base task interface
│   │   └── humanoid_walking.py # Humanoid walking task
│   ├── training/          # Training pipelines
│   │   └── curriculum_trainer.py # Curriculum training
│   ├── rl/                # Reinforcement learning components
│   │   └── ppo_bridge.py  # PPO integration bridge
│   └── utils/             # Utilities
│       └── model_factory.py # Factory functions
├── assets/                # MuJoCo models and assets
├── main.py               # Training entry point
└── requirements.txt      # Dependencies
```

## Installation

1. Create and activate the conda environment:
```bash
conda create -n mujoco_env python=3.9
conda activate mujoco_env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install MuJoCo Menagerie for Unitree H1 robot:
```bash
git clone https://github.com/deepmind/mujoco_menagerie.git ~/packages/mujoco_menagerie
```

## Usage

### Training

Run training with default configuration (uses Unitree H1 robot):
```bash
python main.py
```

Enable MuJoCo viewer for visualization:
```bash
python main.py task.mujoco.viewer_enabled=true
```

Override configuration parameters:
```bash
python main.py model.joint_units=512 training.phases.phase_1.duration_iterations=20000
```

Use different configurations:
```bash
python main.py --config-path config --config-name my_config
```

### Testing

Run evaluation on trained model:
```bash
python test.py checkpoints/latest.pt --episodes 50 --output-dir results
```

View evaluation plots and metrics:
```bash
python test.py checkpoints/latest.pt --episodes 10 --output-dir results
# Check results/plots.png and results/results.json
```

### Configuration

The system uses Hydra for configuration management. Key configuration files:

- `config/config.yaml`: Main configuration with defaults
- `config/model/donet_gru.yaml`: Model architecture parameters
- `config/task/humanoid_walking.yaml`: Task-specific settings
- `config/training/iterative_curriculum.yaml`: Training curriculum

## Key Features

### 1. Donet GRU Architecture

The `DonetGRU` model implements specialized unit groups:
- **Joint Units**: Process DoG-encoded joint angles
- **Perceptual Units**: Handle MuJoCo sensor data
- **Place Units**: Associate configurations with contexts
- **Planning Units**: Generate action sequences for internal simulation
- **Torque Units**: Output joint torque commands

### 2. Curriculum Training

Three training phases:
1. **Pure Exploration** (10k iterations): High noise, short trajectories, random goals
2. **Noise Reduction** (30k iterations): Annealing noise, structured goals, planning loss
3. **Refined Control** (60k iterations): Minimal noise, multi-step planning

### 3. Task Modularity

The `BaseTask` interface allows extending to other robotics problems:
- Environment setup and simulation
- Observation and action spaces
- Goal generation strategies
- Reward computation

### 4. PPO RL Bridge

The system includes **Proximal Policy Optimization (PPO)** integration:

**Training Pipeline**:
1. **Donet Encoder Training** (0-50k iterations): Learn robot dynamics through curriculum
2. **PPO + Donet Execution** (50k+ iterations): PPO provides high-level walking strategy

**PPO Architecture**:
- **State**: High-level walking metrics (velocity, balance, energy, contact state)
- **Actions**: Walking context signals (velocity targets, balance cues, energy conservation) 
- **Rewards**: Walking performance (forward progress, balance, efficiency)
- **Integration**: PPO actions → Donet planning goals → low-level motor control

**Configuration**:
```bash
# Enable PPO (default: enabled)
python main.py rl.use_ppo=true rl.ppo_start_iteration=50000

# Disable PPO (Donet-only training)
python main.py rl.use_ppo=false
```

### 5. Difference of Gaussians Encoding

Joint angles are encoded using DoG tuning curves for sparse, biologically-inspired representations with lateral inhibition.

## Extending to New Tasks

To add a new robotics task:

1. Create a new task class inheriting from `BaseTask`
2. Add task configuration in `config/task/`
3. Register the task in `model_factory.py`
4. Run with your task configuration

Example:
```python
class QuadrupedWalkingTask(BaseTask):
    def __init__(self, config):
        # Implement task-specific methods
        pass
```

## Development Status

This is a development framework implementing the core architecture. Key components implemented:

- ✅ GRU architecture with specialized units
- ✅ DoG joint encoding
- ✅ Task interface and humanoid walking task
- ✅ Curriculum training framework
- ✅ PPO RL bridge integration
- ✅ Hydra configuration system

TODO items (marked in code):
- Input masking and reconstruction loss implementation
- Planning forward pass and goal reaching
- Validation metrics and episodes
- MuJoCo integration (currently has placeholders)
- RL bridge for high-level walking objectives

## Citations

This implementation is based on the REMI architecture and neuroscience principles described in the project documentation (CLAUDE.md).