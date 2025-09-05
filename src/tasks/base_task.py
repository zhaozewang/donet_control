from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional
import mujoco


class BaseTask(ABC):
    """
    Base class for all tasks that can be solved with REMI.
    
    Tasks define the environment, sensors, rewards, and goal generation.
    This allows extending to different robotics problems beyond humanoid walking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base task with configuration.
        
        Args:
            config: Task-specific configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Will be set by subclasses
        self.env = None
        self.n_joints = None
        self.percept_dim = None
        
    @abstractmethod
    def setup_environment(self) -> None:
        """Setup the MuJoCo environment."""
        pass
        
    @abstractmethod
    def get_observation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current observation from environment.
        
        Returns:
            joint_angles: [n_joints] current joint angles
            perceptual_data: [percept_dim] sensor data
        """
        pass
        
    @abstractmethod
    def apply_torques(self, torques: torch.Tensor) -> None:
        """
        Apply torque commands to environment.
        
        Args:
            torques: [n_joints] torque commands
        """
        pass
        
    @abstractmethod
    def step_simulation(self) -> None:
        """Step the physics simulation forward."""
        pass
        
    @abstractmethod
    def generate_random_goal(self) -> torch.Tensor:
        """
        Generate a random goal for exploration.
        
        Returns:
            goal: Goal state tensor
        """
        pass
        
    @abstractmethod
    def generate_structured_goal(self, difficulty: float = 0.5) -> torch.Tensor:
        """
        Generate a structured goal for curriculum learning.
        
        Args:
            difficulty: Difficulty level [0, 1]
            
        Returns:
            goal: Goal state tensor
        """
        pass
        
    @abstractmethod
    def compute_reward(
        self, 
        state: Dict[str, torch.Tensor], 
        action: torch.Tensor,
        next_state: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute reward for RL component.
        
        Args:
            state: Current state dictionary
            action: Action taken
            next_state: Resulting state dictionary
            
        Returns:
            reward: Scalar reward value
        """
        pass
        
    @abstractmethod
    def is_done(self) -> bool:
        """Check if episode is finished."""
        pass
        
    @abstractmethod
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset environment to initial state.
        
        Returns:
            joint_angles: Initial joint angles
            perceptual_data: Initial perceptual data
        """
        pass
        
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get complete state information for RL/planning."""
        joint_angles, perceptual_data = self.get_observation()
        return {
            'joint_angles': joint_angles,
            'perceptual_data': perceptual_data,
            'timestamp': torch.tensor([self.get_time()], device=self.device)
        }
    
    def get_time(self) -> float:
        """Get current simulation time."""
        if self.env is not None and hasattr(self.env, 'data'):
            return self.env.data.time
        return 0.0
    
    @property
    def observation_dims(self) -> Dict[str, int]:
        """Get observation dimensions for model initialization."""
        return {
            'n_joints': self.n_joints,
            'percept_dim': self.percept_dim
        }