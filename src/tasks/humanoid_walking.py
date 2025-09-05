import torch
import numpy as np
import mujoco
from typing import Dict, Tuple, Any, Optional
from .base_task import BaseTask


class HumanoidWalkingTask(BaseTask):
    """
    Humanoid walking task in MuJoCo environment.
    
    Implements the specific sensors, rewards, and goal generation
    for teaching a humanoid robot to walk.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract configuration
        self.mujoco_config = config['mujoco']
        self.robot_config = config['robot'] 
        self.sensor_config = config['sensors']
        self.walking_config = config['walking']
        self.reward_config = config['rewards']
        
        # Robot parameters
        self.n_joints = self.robot_config['n_joints']
        self.percept_dim = self.sensor_config['total_percept_dim']
        self.joint_names = self.robot_config['joint_names']
        
        # Walking parameters
        self.target_velocity = self.walking_config['target_velocity']
        self.balance_threshold = self.walking_config['balance_threshold']
        
        # Simulation state
        self.episode_step = 0
        self.max_episode_length = self.mujoco_config['episode_length']
        self.viewer_enabled = self.mujoco_config.get('viewer_enabled', False)
        self.viewer = None
        
        # Setup environment
        self.setup_environment()
        
    def setup_environment(self) -> None:
        """Setup MuJoCo humanoid environment."""
        # Load MuJoCo model (expand home directory path)
        xml_path = self.mujoco_config['xml_path']
        if xml_path.startswith('~'):
            from pathlib import Path
            xml_path = str(Path(xml_path).expanduser())
            
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            print(f"✓ Successfully loaded Unitree H1 model from {xml_path}")
        except Exception as e:
            # Fallback to placeholder for development
            print(f"Warning: Could not load MuJoCo model from {xml_path}: {e}")
            print("Creating placeholder environment for development")
            print("To use Unitree H1, install mujoco_menagerie:")
            print("  git clone https://github.com/deepmind/mujoco_menagerie.git ~/packages/mujoco_menagerie")
            self.model = None
            self.data = None
            
        # Set simulation parameters
        if self.model:
            self.model.opt.timestep = self.mujoco_config['timestep']
            
            # Setup viewer if requested
            if self.viewer_enabled:
                try:
                    import mujoco.viewer
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    print("✓ MuJoCo viewer launched")
                except Exception as e:
                    print(f"Warning: Could not launch viewer: {e}")
                    self.viewer = None
            
        # Initialize joint mappings
        self._setup_joint_mappings()
        
    def _setup_joint_mappings(self) -> None:
        """Setup mappings between joint names and indices."""
        if self.model is None:
            # Placeholder mappings
            self.joint_indices = {name: i for i, name in enumerate(self.joint_names)}
        else:
            self.joint_indices = {}
            for name in self.joint_names:
                try:
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    self.joint_indices[name] = joint_id
                except:
                    print(f"Warning: Joint {name} not found in model")
                    
    def get_observation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current observation from MuJoCo simulation."""
        if self.data is None:
            # Return placeholder observations for development
            joint_angles = torch.zeros(self.n_joints, device=self.device)
            perceptual_data = torch.zeros(self.percept_dim, device=self.device)
            return joint_angles, perceptual_data
            
        # Extract joint angles
        joint_angles = torch.tensor(
            self.data.qpos[self._get_joint_qpos_indices()],
            dtype=torch.float32, device=self.device
        )
        
        # Extract perceptual data
        perceptual_data = self._extract_perceptual_data()
        
        return joint_angles, perceptual_data
        
    def _get_joint_qpos_indices(self) -> np.ndarray:
        """Get qpos indices for controlled joints."""
        if self.model is None:
            return np.arange(self.n_joints)
            
        indices = []
        for name in self.joint_names:
            if name in self.joint_indices:
                joint_id = self.joint_indices[name]
                qpos_addr = self.model.jnt_qposadr[joint_id]
                indices.append(qpos_addr)
        return np.array(indices)
        
    def _extract_perceptual_data(self) -> torch.Tensor:
        """Extract all perceptual sensor data."""
        if self.data is None:
            return torch.zeros(self.percept_dim, device=self.device)
            
        sensor_data = []
        
        # IMU data (accelerometer + gyroscope)
        imu_data = self._get_imu_data()
        sensor_data.append(imu_data)
        
        # Contact forces
        contact_data = self._get_contact_forces()
        sensor_data.append(contact_data)
        
        # Joint torques and velocities
        joint_torques = torch.tensor(
            self.data.ctrl[:self.n_joints], 
            dtype=torch.float32, device=self.device
        )
        joint_velocities = torch.tensor(
            self.data.qvel[self._get_joint_qvel_indices()],
            dtype=torch.float32, device=self.device  
        )
        sensor_data.extend([joint_torques, joint_velocities])
        
        # Center of mass dynamics
        com_data = self._get_com_data()
        sensor_data.append(com_data)
        
        # Additional miscellaneous sensors
        misc_data = self._get_misc_sensors()
        sensor_data.append(misc_data)
        
        return torch.cat(sensor_data)
        
    def _get_imu_data(self) -> torch.Tensor:
        """Extract IMU sensor data (accelerometer + gyroscope)."""
        # Placeholder - would extract from MuJoCo sensors
        return torch.zeros(self.sensor_config['imu_dim'], device=self.device)
        
    def _get_contact_forces(self) -> torch.Tensor:
        """Extract contact force data."""
        # Placeholder - would extract foot contact forces
        return torch.zeros(self.sensor_config['contact_dim'], device=self.device)
        
    def _get_joint_qvel_indices(self) -> np.ndarray:
        """Get qvel indices for controlled joints."""
        # Placeholder - would map joint names to qvel indices
        return np.arange(self.n_joints)
        
    def _get_com_data(self) -> torch.Tensor:
        """Extract center of mass position and velocity."""
        # Placeholder - would compute from MuJoCo data
        return torch.zeros(self.sensor_config['com_dynamics_dim'], device=self.device)
        
    def _get_misc_sensors(self) -> torch.Tensor:
        """Extract miscellaneous sensor data."""
        # Placeholder - would include body positions, orientations, etc.
        return torch.zeros(self.sensor_config['misc_dim'], device=self.device)
        
    def apply_torques(self, torques: torch.Tensor) -> None:
        """Apply torque commands to MuJoCo simulation."""
        if self.data is None:
            return
            
        # Clamp torques to limits
        torque_limits = self.robot_config['torque_limits']
        clamped_torques = torch.clamp(
            torques, 
            min=torque_limits[0], 
            max=torque_limits[1]
        )
        
        # Apply to simulation
        self.data.ctrl[:self.n_joints] = clamped_torques.cpu().numpy()
        
    def step_simulation(self) -> None:
        """Step MuJoCo physics simulation."""
        if self.model is None or self.data is None:
            return
            
        # Step simulation with frame skipping
        for _ in range(self.mujoco_config['frame_skip']):
            mujoco.mj_step(self.model, self.data)
            
        # Update viewer if active
        if self.viewer is not None:
            try:
                self.viewer.sync()
            except:
                # Viewer might have been closed
                self.viewer = None
            
        self.episode_step += 1
        
    def generate_random_goal(self) -> torch.Tensor:
        """Generate random joint configuration goal."""
        joint_limits = self.robot_config['joint_limits']
        random_config = torch.uniform(
            joint_limits[0], joint_limits[1], 
            size=(self.n_joints,), device=self.device
        )
        return random_config
        
    def generate_structured_goal(self, difficulty: float = 0.5) -> torch.Tensor:
        """Generate structured walking goal based on difficulty."""
        # For walking, goals could be forward velocity targets,
        # specific poses, or gait parameters
        
        # Simple example: scale target velocity by difficulty
        scaled_velocity = self.target_velocity * difficulty
        
        # Convert to joint space goal (placeholder)
        # In reality, this would use inverse kinematics or learned mappings
        goal_config = torch.zeros(self.n_joints, device=self.device)
        
        # Add some variation based on walking cycle
        time_factor = (self.episode_step * self.model.opt.timestep) if self.model else 0.0
        for i in range(self.n_joints):
            goal_config[i] = 0.2 * torch.sin(torch.tensor(time_factor + i))
            
        return goal_config
        
    def compute_reward(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor, 
        next_state: Dict[str, torch.Tensor]
    ) -> float:
        """Compute walking reward."""
        reward = 0.0
        
        # Forward progress reward
        if 'com_velocity' in next_state:
            forward_vel = next_state['com_velocity'][0]  # x-velocity
            reward += self.reward_config['forward_progress'] * forward_vel
            
        # Balance maintenance reward
        if 'orientation' in next_state:
            tilt = torch.abs(next_state['orientation'][:2]).sum()  # Roll + pitch
            balance_reward = torch.exp(-tilt / self.balance_threshold)
            reward += self.reward_config['balance_maintenance'] * balance_reward
            
        # Energy efficiency penalty
        energy_cost = torch.sum(action ** 2)
        reward -= self.reward_config['energy_efficiency'] * energy_cost
        
        # Smoothness reward (penalize large torque changes)
        if hasattr(self, 'prev_action'):
            smoothness = torch.sum((action - self.prev_action) ** 2)
            reward -= self.reward_config['smooth_gait'] * smoothness
        self.prev_action = action.clone()
        
        # Alive bonus
        reward += self.reward_config['alive_bonus']
        
        return reward.item()
        
    def is_done(self) -> bool:
        """Check if episode should terminate."""
        # Episode length limit
        if self.episode_step >= self.max_episode_length:
            return True
            
        # Fall detection (placeholder)
        if self.data is not None:
            # Check if torso height is too low
            torso_height = self.data.qpos[2] if len(self.data.qpos) > 2 else 1.0
            if torso_height < 0.5:  # Approximate fall threshold
                return True
                
        return False
        
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset environment to initial state."""
        self.episode_step = 0
        
        if self.data is not None:
            # Reset MuJoCo simulation
            mujoco.mj_resetData(self.model, self.data)
            
            # Add small random perturbations to initial state
            noise_scale = 0.02
            self.data.qpos += np.random.normal(0, noise_scale, self.data.qpos.shape)
            self.data.qvel += np.random.normal(0, noise_scale, self.data.qvel.shape)
            
        # Reset previous action for smoothness reward
        self.prev_action = torch.zeros(self.n_joints, device=self.device)
        
        return self.get_observation()