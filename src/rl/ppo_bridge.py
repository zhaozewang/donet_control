import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces

from ..models.donet_gru import DonetGRU
from ..tasks.base_task import BaseTask


class DonetFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses Donet place units as high-level state representation.
    
    This extracts the place units from the Donet model to provide a rich,
    learned representation of the robot's current situation for PPO.
    """
    
    def __init__(
        self, 
        observation_space: gym.Space,
        donet_model: DonetGRU,
        features_dim: int = 512
    ):
        # Use place units as feature dimension
        super().__init__(observation_space, remi_model.place_units)
        
        self.donet_model = donet_model
        self._features_dim = remi_model.place_units
        
        # Freeze Donet parameters - we don't want PPO to modify the encoder
        for param in self.donet_model.parameters():
            param.requires_grad = False
            
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract place unit features from Donet model.
        
        Args:
            observations: [batch, obs_dim] raw observations
            
        Returns:
            features: [batch, place_units] place unit activations
        """
        # Observations contain [joint_angles, perceptual_data]
        batch_size = observations.shape[0]
        
        # Split observations (assumes first n_joints are joint angles, rest is perceptual)
        n_joints = self.donet_model.n_joints
        joint_angles = observations[:, :n_joints]
        perceptual_data = observations[:, n_joints:]
        
        # Get Donet hidden state (encoding mode)
        with torch.no_grad():  # Don't update Donet during PPO training
            hidden = self.donet_model.init_hidden(batch_size, observations.device)
            hidden, _ = self.donet_model(
                joint_angles=joint_angles,
                perceptual_inputs=perceptual_data,
                hidden=hidden,
                mode="encoding"
            )
            
            # Extract place units
            unit_groups = self.donet_model.gru_cell.get_unit_groups(hidden)
            place_features = unit_groups['place']
            
        return place_features


class PPOBridge(nn.Module):
    """
    PPO bridge that provides high-level walking guidance to the Donet system.
    
    The PPO agent operates on:
    - State: High-level walking metrics from place units
    - Actions: Walking context signals (velocity targets, balance cues, etc.)
    - Rewards: Walking performance (forward progress, balance, efficiency)
    
    The Donet system handles all low-level motor control and planning.
    """
    
    def __init__(
        self,
        donet_model: DonetGRU,
        task: BaseTask,
        config: Dict[str, Any]
    ):
        super().__init__()
        
        self.donet_model = donet_model
        self.task = task
        self.config = config
        
        self.device = next(remi_model.parameters()).device
        
        # Define observation and action spaces for PPO
        self._setup_spaces()
        
        # Create PPO agent
        self._setup_ppo_agent()
        
        # Training state
        self.episode_rewards = []
        self.episode_steps = 0
        self.max_episode_steps = config.get('max_episode_steps', 1000)
        
    def _setup_spaces(self):
        """Setup observation and action spaces for PPO."""
        # Observation space: high-level walking state
        # [forward_velocity, lateral_velocity, angular_velocity, 
        #  balance_angle_x, balance_angle_y, height, energy_rate, contact_state]
        obs_dim = 8
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: walking context signals
        # [target_forward_vel, target_lateral_vel, balance_urgency, energy_conservation]
        action_dim = 4
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
    def _setup_ppo_agent(self):
        """Initialize PPO agent with custom feature extractor."""
        # Custom policy using Donet features
        policy_kwargs = {
            "features_extractor_class": DonetFeatureExtractor,
            "features_extractor_kwargs": {
                "donet_model": self.donet_model,
                "features_dim": self.donet_model.place_units
            },
            "net_arch": dict(
                pi=[256, 256],  # Policy network
                vf=[256, 256]   # Value function network
            )
        }
        
        self.ppo_agent = PPO(
            policy="MlpPolicy",
            env=None,  # We'll handle environment manually
            learning_rate=self.config.get('learning_rate', 3e-4),
            n_steps=self.config.get('n_steps', 2048),
            batch_size=self.config.get('batch_size', 64),
            n_epochs=self.config.get('n_epochs', 10),
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 0.95),
            clip_range=self.config.get('clip_range', 0.2),
            ent_coef=self.config.get('ent_coef', 0.01),
            vf_coef=self.config.get('vf_coef', 0.5),
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.device
        )
        
    def get_high_level_observation(self) -> np.ndarray:
        """
        Extract high-level walking state for PPO.
        
        Returns:
            obs: [obs_dim] high-level walking metrics
        """
        # Get current robot state
        state_dict = self.task.get_state_dict()
        
        # Extract high-level metrics (placeholder implementation)
        obs = np.zeros(8, dtype=np.float32)
        
        # Forward velocity (from CoM or computed)
        obs[0] = 0.0  # TODO: Extract actual forward velocity
        
        # Lateral velocity
        obs[1] = 0.0  # TODO: Extract actual lateral velocity
        
        # Angular velocity (yaw rate)
        obs[2] = 0.0  # TODO: Extract actual angular velocity
        
        # Balance angles (roll, pitch)
        obs[3] = 0.0  # TODO: Extract roll angle
        obs[4] = 0.0  # TODO: Extract pitch angle
        
        # Height (CoM height)
        obs[5] = 1.0  # TODO: Extract actual height
        
        # Energy rate (power consumption)
        obs[6] = 0.0  # TODO: Compute energy consumption rate
        
        # Contact state (0=flight, 1=single support, 2=double support)
        obs[7] = 1.0  # TODO: Extract actual contact state
        
        return obs
        
    def convert_action_to_goals(self, action: np.ndarray) -> torch.Tensor:
        """
        Convert PPO action to Donet planning goals.
        
        Args:
            action: [4] PPO action [target_vel_x, target_vel_y, balance, energy]
            
        Returns:
            planning_goals: [planning_units] goals for Donet planning units
        """
        # Create planning goal vector
        planning_goals = torch.zeros(
            self.donet_model.planning_units, 
            device=self.device
        )
        
        # Map high-level actions to planning goals
        # This is a simple mapping - could be learned or more sophisticated
        
        # Forward velocity target -> first set of planning units
        target_vel_x = action[0] * 2.0  # Scale to reasonable velocity
        planning_goals[:64] = target_vel_x * 0.1  # Broadcast to first 64 units
        
        # Lateral velocity target -> second set
        target_vel_y = action[1] * 1.0
        planning_goals[64:128] = target_vel_y * 0.1
        
        # Balance urgency -> third set
        balance_signal = action[2]
        planning_goals[128:192] = balance_signal * 0.2
        
        # Energy conservation -> remaining units
        energy_signal = action[3]
        planning_goals[192:] = energy_signal * 0.1
        
        return planning_goals.unsqueeze(0)  # Add batch dimension
        
    def compute_reward(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: np.ndarray,
        current_state: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute walking reward for PPO training.
        
        Args:
            prev_state: Previous state dictionary
            action: PPO action taken
            current_state: Current state dictionary
            
        Returns:
            reward: Scalar reward value
        """
        reward = 0.0
        
        # Forward progress reward
        target_vel_x = action[0] * 2.0
        actual_vel_x = 0.0  # TODO: Extract from state
        vel_reward = -abs(actual_vel_x - target_vel_x)
        reward += vel_reward
        
        # Balance reward
        balance_angle = abs(0.0)  # TODO: Extract tilt from state
        balance_reward = np.exp(-balance_angle * 2.0)  # Exponential penalty for tilt
        reward += balance_reward
        
        # Energy efficiency
        energy_cost = 0.0  # TODO: Compute from torques
        reward -= 0.01 * energy_cost
        
        # Alive bonus
        reward += 0.1
        
        return reward
        
    def step(
        self, 
        donet_hidden: torch.Tensor,
        obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, np.ndarray, float, bool, Dict]:
        """
        Single step of PPO-guided Donet execution.
        
        Args:
            donet_hidden: Current Donet hidden state
            obs: Current observations
            
        Returns:
            new_hidden: Updated Donet hidden state
            ppo_action: PPO action taken
            reward: Step reward
            done: Whether episode is complete
            info: Additional information
        """
        # Get high-level observation for PPO
        ppo_obs = self.get_high_level_observation()
        
        # Get PPO action
        ppo_action, _ = self.ppo_agent.predict(ppo_obs, deterministic=False)
        
        # Convert to Donet planning goals
        planning_goals = self.convert_action_to_goals(ppo_action)
        
        # Execute Donet planning step
        new_hidden, torque_commands = self.donet_model(
            planning_goals=planning_goals,
            hidden=donet_hidden,
            mode="planning"
        )
        
        # Apply torques to environment
        self.task.apply_torques(torque_commands.squeeze(0))
        self.task.step_simulation()
        
        # Get new observation and compute reward
        new_obs = self.task.get_state_dict()
        reward = self.compute_reward(obs, ppo_action, new_obs)
        
        # Check if done
        self.episode_steps += 1
        done = (
            self.episode_steps >= self.max_episode_steps or 
            self.task.is_done()
        )
        
        # Store for PPO training
        self.episode_rewards.append(reward)
        
        info = {
            'ppo_action': ppo_action,
            'planning_goals': planning_goals.cpu().numpy(),
            'torque_commands': torque_commands.cpu().numpy(),
            'episode_steps': self.episode_steps
        }
        
        return new_hidden, ppo_action, reward, done, info
        
    def update_ppo(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray], 
        rewards: List[float],
        dones: List[bool]
    ):
        """Update PPO policy using collected experience."""
        if len(observations) < self.config.get('min_update_size', 100):
            return
            
        # Convert to arrays
        obs_array = np.array(observations)
        action_array = np.array(actions)
        reward_array = np.array(rewards)
        done_array = np.array(dones)
        
        # Create experience buffer (simplified)
        # In a full implementation, you'd use PPO's rollout buffer
        
        # For now, just log the performance
        mean_reward = np.mean(reward_array)
        print(f"PPO Update - Mean Reward: {mean_reward:.3f}, Episodes: {len(observations)}")
        
    def reset_episode(self):
        """Reset for new episode."""
        self.episode_rewards = []
        self.episode_steps = 0
        
    def save(self, path: str):
        """Save PPO model."""
        self.ppo_agent.save(path)
        
    def load(self, path: str):
        """Load PPO model."""
        self.ppo_agent = PPO.load(path, device=self.device)