import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import wandb
from omegaconf import DictConfig
import logging
from pathlib import Path

from ..models.donet_gru import DonetGRU
from ..tasks.base_task import BaseTask
from ..rl.ppo_bridge import PPOBridge


class CurriculumTrainer:
    """
    Implements the 3-phase curriculum training from CLAUDE.md:
    1. Pure exploration with high noise
    2. Gradual noise reduction with goal-directed training  
    3. Refined goal-directed movement with minimal noise
    """
    
    def __init__(
        self, 
        model: DonetGRU,
        task: BaseTask,
        config: DictConfig
    ):
        self.model = model
        self.task = task
        self.config = config
        
        self.device = torch.device(config.get('device', 'cpu'))
        self.model.to(self.device)
        
        # Training state
        self.current_iteration = 0
        self.current_phase = 1
        self.phase_start_iteration = 0
        
        # Setup optimizer and scheduler
        self._setup_optimization()
        
        # Setup logging
        self._setup_logging()
        
        # Phase configurations
        self.phases = config.phases
        self.phase_names = ['phase_1', 'phase_2', 'phase_3']
        
        # Current phase config
        self.current_phase_config = self.phases[self.phase_names[0]]
        
        # Setup PPO bridge if enabled
        self.ppo_bridge = None
        self.use_ppo = config.get('rl', {}).get('use_ppo', False)
        self.ppo_start_iteration = config.get('rl', {}).get('ppo_start_iteration', 50000)
        
        if self.use_ppo:
            print("PPO integration enabled - will start after Donet encoder training")
        
        # Training statistics
        self.stats = {
            'reconstruction_loss': [],
            'planning_loss': [],
            'goal_reaching_success': [],
            'trajectory_smoothness': [],
            'ppo_reward': []
        }
        
    def _setup_optimization(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        optimizer_config = self.config.optimizer
        
        if optimizer_config.type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.weight_decay,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.type}")
            
        # Setup scheduler
        scheduler_config = self.config.scheduler
        if scheduler_config.type.lower() == 'cosine_annealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.T_max,
                eta_min=scheduler_config.eta_min
            )
        else:
            self.scheduler = None
            
    def _setup_logging(self) -> None:
        """Setup logging and checkpointing."""
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.log_dir = Path(self.config.logging.log_dir)
        self.checkpoint_dir = Path(self.config.logging.checkpoint_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup wandb if enabled
        if self.config.logging.wandb.enabled:
            wandb.init(
                project=self.config.logging.wandb.project,
                entity=self.config.logging.wandb.entity,
                name=self.config.logging.wandb.name,
                config=dict(self.config)
            )
            
    def train(self) -> None:
        """Main training loop."""
        self.logger.info("Starting curriculum training")
        
        while self.current_iteration < self.config.global.max_iterations:
            # Check if we need to advance to next phase
            self._maybe_advance_phase()
            
            # Initialize PPO if we've reached the start iteration
            self._maybe_init_ppo()
            
            # Run training iteration (REMI-only or REMI+PPO)
            if self.ppo_bridge is not None:
                losses = self._training_iteration_with_ppo()
            else:
                losses = self._training_iteration()
            
            # Log and save
            if self.current_iteration % self.config.logging.log_frequency == 0:
                self._log_metrics(losses)
                
            if self.current_iteration % self.config.logging.save_frequency == 0:
                self._save_checkpoint()
                
            # Validation
            if self.current_iteration % self.config.validation.frequency == 0:
                self._validate()
                
            self.current_iteration += 1
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                
        self.logger.info("Training completed")
        
    def _maybe_advance_phase(self) -> None:
        """Check if we should advance to the next training phase."""
        phase_duration = self.current_phase_config.duration_iterations
        iterations_in_phase = self.current_iteration - self.phase_start_iteration
        
        if iterations_in_phase >= phase_duration and self.current_phase < 3:
            self.current_phase += 1
            self.phase_start_iteration = self.current_iteration
            self.current_phase_config = self.phases[self.phase_names[self.current_phase - 1]]
            
            self.logger.info(
                f"Advancing to {self.current_phase_config.name} "
                f"at iteration {self.current_iteration}"
            )
    
    def _maybe_init_ppo(self) -> None:
        """Initialize PPO bridge when we reach the start iteration."""
        if (self.use_ppo and 
            self.ppo_bridge is None and 
            self.current_iteration >= self.ppo_start_iteration):
            
            self.logger.info(f"Initializing PPO bridge at iteration {self.current_iteration}")
            self.ppo_bridge = PPOBridge(
                remi_model=self.model,
                task=self.task,
                config=self.config.rl
            )
            self.logger.info("PPO bridge initialized - switching to intertwined planning-execution")
    
    def _training_iteration_with_ppo(self) -> Dict[str, float]:
        """Training iteration with PPO-guided Donet execution."""
        self.model.eval()  # Donet is frozen during PPO execution
        
        # Run episodes with PPO guidance
        episode_rewards = []
        episode_observations = []
        episode_actions = []
        episode_dones = []
        
        n_episodes = self.config.global.get('ppo_episodes_per_iteration', 4)
        
        for episode in range(n_episodes):
            # Reset environment and PPO episode
            initial_joints, initial_percept = self.task.reset()
            self.ppo_bridge.reset_episode()
            
            # Initialize Donet hidden state
            hidden = self.model.init_hidden(1, self.device)
            
            episode_reward = 0.0
            observations = []
            actions = []
            rewards = []
            dones = []
            
            # Run episode with intertwined planning-execution
            for step in range(self.ppo_bridge.max_episode_steps):
                # Get current state
                current_obs = self.task.get_state_dict()
                
                # PPO-guided Donet step
                hidden, ppo_action, reward, done, info = self.ppo_bridge.step(
                    hidden, current_obs
                )
                
                # Store experience
                ppo_obs = self.ppo_bridge.get_high_level_observation()
                observations.append(ppo_obs)
                actions.append(ppo_action)
                rewards.append(reward)
                dones.append(done)
                
                episode_reward += reward
                
                if done:
                    break
            
            # Store episode data
            episode_rewards.append(episode_reward)
            episode_observations.extend(observations)
            episode_actions.extend(actions)
            episode_dones.extend(dones)
        
        # Update PPO policy
        if len(episode_observations) > 0:
            self.ppo_bridge.update_ppo(
                episode_observations, episode_actions, 
                [r for ep_rewards in [rewards] for r in ep_rewards],  # Flatten rewards
                episode_dones
            )
        
        # Return loss statistics
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        return {
            'ppo_mean_reward': mean_reward,
            'ppo_episodes': len(episode_rewards),
            'reconstruction_loss': 0.0,  # No Donet training during PPO phase
            'planning_loss': 0.0
        }
            
    def _training_iteration(self) -> Dict[str, float]:
        """Single training iteration."""
        self.model.train()
        
        # Generate training batch
        batch_data = self._generate_training_batch()
        
        # Forward pass with encoding
        encoding_losses = self._encoding_forward_pass(batch_data)
        
        # Forward pass with planning (if applicable)
        planning_losses = self._planning_forward_pass(batch_data)
        
        # Combine losses
        total_loss = self._combine_losses(encoding_losses, planning_losses)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if hasattr(self.config.global, 'grad_clip_norm'):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.global.grad_clip_norm
            )
            
        self.optimizer.step()
        
        # Return loss statistics
        losses = {**encoding_losses, **planning_losses, 'total_loss': total_loss.item()}
        return losses
        
    def _generate_training_batch(self) -> Dict[str, torch.Tensor]:
        """Generate a batch of training data based on current phase."""
        batch_size = self.config.global.batch_size
        batch_data = {}
        
        # Generate trajectories of varying length
        traj_length_range = self.current_phase_config.trajectory.length
        traj_length = np.random.randint(traj_length_range[0], traj_length_range[1] + 1)
        
        # Initialize batch storage
        joint_angles_seq = []
        perceptual_seq = []
        torque_seq = []
        goals = []
        
        for b in range(batch_size):
            # Reset environment for each trajectory
            initial_joints, initial_percept = self.task.reset()
            
            # Generate goal based on phase
            if self.current_phase_config.goals.type == "random_joint_configs":
                goal = self.task.generate_random_goal()
            else:
                difficulty = np.random.uniform(0.2, 1.0)
                goal = self.task.generate_structured_goal(difficulty)
                
            goals.append(goal)
            
            # Generate trajectory
            traj_joints = [initial_joints]
            traj_percept = [initial_percept] 
            traj_torques = []
            
            hidden = self.model.init_hidden(1, self.device)
            
            for t in range(traj_length):
                # Get current observation
                joints = traj_joints[-1].unsqueeze(0)  # [1, n_joints]
                percept = traj_percept[-1].unsqueeze(0)  # [1, percept_dim]
                
                # Forward pass to get torque commands
                hidden, torques = self.model(
                    joint_angles=joints,
                    perceptual_inputs=percept,
                    hidden=hidden,
                    mode="encoding"
                )
                
                # Add noise based on phase
                noise_std = self._get_current_noise_std()
                noisy_torques = torques + torch.randn_like(torques) * noise_std
                
                # Apply torques and step simulation
                self.task.apply_torques(noisy_torques.squeeze(0))
                self.task.step_simulation()
                
                # Get next observation
                next_joints, next_percept = self.task.get_observation()
                
                # Store data
                traj_joints.append(next_joints)
                traj_percept.append(next_percept)
                traj_torques.append(torques.squeeze(0))
                
            # Convert to tensors
            joint_angles_seq.append(torch.stack(traj_joints[:-1]))  # Exclude final state
            perceptual_seq.append(torch.stack(traj_percept[:-1]))
            torque_seq.append(torch.stack(traj_torques))
            
        # Stack batch data
        # Pad sequences to same length for batching
        max_len = max(len(seq) for seq in joint_angles_seq)
        
        batch_joints = torch.zeros(batch_size, max_len, self.task.n_joints, device=self.device)
        batch_percept = torch.zeros(batch_size, max_len, self.task.percept_dim, device=self.device)
        batch_torques = torch.zeros(batch_size, max_len, self.task.n_joints, device=self.device)
        batch_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        for i, (joints, percept, torques) in enumerate(zip(joint_angles_seq, perceptual_seq, torque_seq)):
            length = len(joints)
            batch_joints[i, :length] = joints
            batch_percept[i, :length] = percept
            batch_torques[i, :length] = torques
            batch_lengths[i] = length
            
        batch_data = {
            'joint_angles': batch_joints,
            'perceptual_inputs': batch_percept,
            'torque_commands': batch_torques,
            'sequence_lengths': batch_lengths,
            'goals': torch.stack(goals)
        }
        
        return batch_data
        
    def _get_current_noise_std(self) -> float:
        """Get current noise standard deviation based on phase and annealing."""
        noise_config = self.current_phase_config.noise
        
        if not noise_config.get('noise_annealing', False):
            return noise_config.get('torque_noise_std', 0.1)
            
        # Exponential annealing: sigma * exp(-alpha * t)
        initial_std = noise_config.torque_noise_std_initial
        final_std = noise_config.torque_noise_std_final
        annealing_rate = noise_config.annealing_rate
        
        iterations_in_phase = self.current_iteration - self.phase_start_iteration
        annealed_std = initial_std * np.exp(-annealing_rate * iterations_in_phase)
        
        return max(annealed_std, final_std)
        
    def _encoding_forward_pass(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Forward pass for encoding phase with input masking."""
        # TODO: Implement encoding forward pass with reconstruction loss
        encoding_losses = {'reconstruction_loss': 0.0}
        return encoding_losses
        
    def _planning_forward_pass(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Forward pass for planning phase."""
        # TODO: Implement planning forward pass with goal reaching
        planning_losses = {'planning_loss': 0.0}
        return planning_losses
        
    def _combine_losses(
        self, 
        encoding_losses: Dict[str, float], 
        planning_losses: Dict[str, float]
    ) -> torch.Tensor:
        """Combine different loss components based on phase."""
        loss_weights = self.current_phase_config.loss
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Reconstruction loss
        if 'reconstruction_loss' in encoding_losses:
            recon_weight = loss_weights.get('reconstruction_weight', 1.0)
            total_loss = total_loss + recon_weight * encoding_losses['reconstruction_loss']
            
        # Planning loss
        if 'planning_loss' in planning_losses:
            planning_weight = loss_weights.get('planning_weight', 0.0)
            total_loss = total_loss + planning_weight * planning_losses['planning_loss']
            
        return total_loss
        
    def _log_metrics(self, losses: Dict[str, float]) -> None:
        """Log training metrics."""
        metrics = {
            'iteration': self.current_iteration,
            'phase': self.current_phase,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'noise_std': self._get_current_noise_std(),
            **losses
        }
        
        # Log to console
        self.logger.info(
            f"Iter {self.current_iteration:6d} | "
            f"Phase {self.current_phase} | "
            f"Loss {losses['total_loss']:.4f}"
        )
        
        # Log to wandb
        if self.config.logging.wandb.enabled:
            wandb.log(metrics, step=self.current_iteration)
            
    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'iteration': self.current_iteration,
            'phase': self.current_phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': dict(self.config)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        checkpoint_path = (
            self.checkpoint_dir / 
            f"checkpoint_iter_{self.current_iteration:06d}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
    def _validate(self) -> None:
        """Run validation episodes."""
        self.model.eval()
        
        with torch.no_grad():
            # TODO: Implement validation episodes
            # - Test goal reaching success rate
            # - Measure trajectory smoothness
            # - Evaluate reconstruction accuracy
            pass
            
        self.model.train()