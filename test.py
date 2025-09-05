#!/usr/bin/env python3
"""
Testing and evaluation script for Donet Humanoid Control System.

This script evaluates the trained model's walking performance:
- Load trained checkpoint
- Run walking episodes 
- Measure walking metrics (velocity, balance, energy efficiency)
- Generate performance reports and visualizations
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple
import json

from src.utils.model_factory import create_model, create_task
from src.rl.ppo_bridge import PPOBridge


class DonetEvaluator:
    """Evaluator for testing Donet walking performance."""
    
    def __init__(self, checkpoint_path: str, config_path: str = "config/config.yaml"):
        """
        Initialize evaluator with trained checkpoint.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.cfg = OmegaConf.load(config_path)
        
        # Setup device
        if self.cfg.environment.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.environment.device)
            
        # Create task and model
        self.task = create_task(self.cfg.task)
        self.model = create_model(self.cfg.model, self.task)
        self.model.to(self.device)
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Setup PPO if available
        self.ppo_bridge = None
        if self.cfg.rl.use_ppo and Path(checkpoint_path.replace('.pt', '_ppo.pt')).exists():
            self.ppo_bridge = PPOBridge(self.model, self.task, self.cfg.rl)
            try:
                self.ppo_bridge.load(checkpoint_path.replace('.pt', '_ppo.pt'))
                self.logger.info("✓ PPO checkpoint loaded")
            except:
                self.logger.warning("⚠️  PPO checkpoint not found, using random PPO policy")
        
        # Evaluation metrics storage
        self.reset_metrics()
        
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.iteration = checkpoint.get('iteration', 0)
        self.phase = checkpoint.get('phase', 0)
        
        self.logger.info(f"✓ Checkpoint loaded from iteration {self.iteration:,}, phase {self.phase}")
        
    def reset_metrics(self):
        """Reset evaluation metrics."""
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'forward_distances': [],
            'forward_velocities': [],
            'balance_scores': [],
            'energy_costs': [],
            'fall_count': 0,
            'success_count': 0
        }
        
    def run_episode(self, max_steps: int = 1000, render: bool = False) -> Dict:
        """
        Run a single evaluation episode.
        
        Args:
            max_steps: Maximum episode steps
            render: Whether to render (placeholder)
            
        Returns:
            episode_info: Dictionary with episode statistics
        """
        # Reset environment
        joint_angles, perceptual_data = self.task.reset()
        
        # Initialize hidden state
        hidden = self.model.init_hidden(1, self.device)
        
        # Episode tracking
        episode_reward = 0.0
        episode_steps = 0
        forward_distance = 0.0
        energy_cost = 0.0
        balance_violations = 0
        
        initial_com_pos = None
        
        # Reset PPO episode if available
        if self.ppo_bridge:
            self.ppo_bridge.reset_episode()
        
        for step in range(max_steps):
            # Get current state
            current_state = self.task.get_state_dict()
            
            if initial_com_pos is None:
                initial_com_pos = 0.0  # TODO: Extract actual initial position
            
            if self.ppo_bridge:
                # PPO-guided execution
                hidden, ppo_action, reward, done, info = self.ppo_bridge.step(
                    hidden, current_state
                )
                torque_commands = info['torque_commands']
            else:
                # Pure Donet execution
                hidden, torque_commands = self.model(
                    joint_angles=joint_angles.unsqueeze(0),
                    perceptual_inputs=perceptual_data.unsqueeze(0),
                    hidden=hidden,
                    mode="encoding"
                )
                
                # Apply torques
                self.task.apply_torques(torque_commands.squeeze(0))
                self.task.step_simulation()
                
                # Compute reward
                next_state = self.task.get_state_dict()
                reward = self.task.compute_reward(current_state, torque_commands.squeeze(0), next_state)
                done = self.task.is_done()
            
            # Update metrics
            episode_reward += reward
            episode_steps += 1
            energy_cost += torch.sum(torque_commands ** 2).item()
            
            # Check balance (placeholder)
            tilt_angle = 0.0  # TODO: Extract actual tilt from state
            if abs(tilt_angle) > 0.5:  # 0.5 radians threshold
                balance_violations += 1
            
            # Get new observation for next step
            joint_angles, perceptual_data = self.task.get_observation()
            
            if done:
                break
        
        # Calculate final metrics
        final_com_pos = 0.0  # TODO: Extract actual final position
        forward_distance = final_com_pos - initial_com_pos
        avg_velocity = forward_distance / (episode_steps * 0.01) if episode_steps > 0 else 0.0  # Assuming 0.01s timestep
        balance_score = 1.0 - (balance_violations / episode_steps) if episode_steps > 0 else 0.0
        
        # Determine success/failure
        fell_down = self.task.is_done() and episode_steps < max_steps
        success = not fell_down and forward_distance > 1.0  # At least 1 meter forward
        
        episode_info = {
            'reward': episode_reward,
            'steps': episode_steps,
            'forward_distance': forward_distance,
            'avg_velocity': avg_velocity,
            'balance_score': balance_score,
            'energy_cost': energy_cost,
            'fell_down': fell_down,
            'success': success
        }
        
        return episode_info
        
    def run_evaluation(self, n_episodes: int = 50, max_steps: int = 1000) -> Dict:
        """
        Run full evaluation across multiple episodes.
        
        Args:
            n_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            
        Returns:
            results: Aggregated evaluation results
        """
        self.logger.info(f"Running evaluation: {n_episodes} episodes, max {max_steps} steps each")
        self.reset_metrics()
        
        self.model.eval()
        
        with torch.no_grad():
            for episode in range(n_episodes):
                self.logger.info(f"Episode {episode + 1}/{n_episodes}")
                
                episode_info = self.run_episode(max_steps)
                
                # Store metrics
                self.metrics['episode_rewards'].append(episode_info['reward'])
                self.metrics['episode_lengths'].append(episode_info['steps'])
                self.metrics['forward_distances'].append(episode_info['forward_distance'])
                self.metrics['forward_velocities'].append(episode_info['avg_velocity'])
                self.metrics['balance_scores'].append(episode_info['balance_score'])
                self.metrics['energy_costs'].append(episode_info['energy_cost'])
                
                if episode_info['fell_down']:
                    self.metrics['fall_count'] += 1
                if episode_info['success']:
                    self.metrics['success_count'] += 1
                    
                # Log episode summary
                self.logger.info(
                    f"  Reward: {episode_info['reward']:.2f}, "
                    f"Steps: {episode_info['steps']}, "
                    f"Distance: {episode_info['forward_distance']:.2f}m, "
                    f"Velocity: {episode_info['avg_velocity']:.2f}m/s, "
                    f"{'Success' if episode_info['success'] else 'Fail'}"
                )
        
        # Compute summary statistics
        results = {
            'n_episodes': n_episodes,
            'success_rate': self.metrics['success_count'] / n_episodes,
            'fall_rate': self.metrics['fall_count'] / n_episodes,
            'mean_reward': np.mean(self.metrics['episode_rewards']),
            'std_reward': np.std(self.metrics['episode_rewards']),
            'mean_distance': np.mean(self.metrics['forward_distances']),
            'std_distance': np.std(self.metrics['forward_distances']),
            'mean_velocity': np.mean(self.metrics['forward_velocities']),
            'std_velocity': np.std(self.metrics['forward_velocities']),
            'mean_balance': np.mean(self.metrics['balance_scores']),
            'mean_energy': np.mean(self.metrics['energy_costs']),
            'mean_episode_length': np.mean(self.metrics['episode_lengths'])
        }
        
        return results
        
    def print_results(self, results: Dict):
        """Print evaluation results summary."""
        self.logger.info("=" * 60)
        self.logger.info("🏃 DONET HUMANOID WALKING EVALUATION RESULTS")
        self.logger.info("=" * 60)
        
        # Success metrics
        self.logger.info(f"Success Rate:      {results['success_rate']:.1%}")
        self.logger.info(f"Fall Rate:         {results['fall_rate']:.1%}")
        self.logger.info(f"Episodes:          {results['n_episodes']}")
        
        self.logger.info("\n📊 Performance Metrics:")
        self.logger.info(f"Reward:            {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        self.logger.info(f"Forward Distance:  {results['mean_distance']:.2f} ± {results['std_distance']:.2f} m")
        self.logger.info(f"Forward Velocity:  {results['mean_velocity']:.2f} ± {results['std_velocity']:.2f} m/s")
        self.logger.info(f"Balance Score:     {results['mean_balance']:.3f}")
        self.logger.info(f"Energy Cost:       {results['mean_energy']:.2f}")
        self.logger.info(f"Episode Length:    {results['mean_episode_length']:.1f} steps")
        
        # Performance assessment
        self.logger.info("\n🎯 Assessment:")
        if results['success_rate'] > 0.8:
            self.logger.info("✅ Excellent walking performance!")
        elif results['success_rate'] > 0.5:
            self.logger.info("✅ Good walking performance")
        elif results['success_rate'] > 0.2:
            self.logger.info("⚠️  Moderate walking performance")
        else:
            self.logger.info("❌ Poor walking performance - needs more training")
            
        self.logger.info("=" * 60)
        
    def plot_results(self, results: Dict, save_path: str = "evaluation_plots.png"):
        """Generate evaluation plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Donet Humanoid Walking Evaluation', fontsize=16)
        
        # Episode rewards
        axes[0, 0].hist(self.metrics['episode_rewards'], bins=20, alpha=0.7)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        
        # Forward distances
        axes[0, 1].hist(self.metrics['forward_distances'], bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Forward Distances')
        axes[0, 1].set_xlabel('Distance (m)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Velocities
        axes[0, 2].hist(self.metrics['forward_velocities'], bins=20, alpha=0.7, color='orange')
        axes[0, 2].set_title('Forward Velocities')
        axes[0, 2].set_xlabel('Velocity (m/s)')
        axes[0, 2].set_ylabel('Frequency')
        
        # Balance scores
        axes[1, 0].hist(self.metrics['balance_scores'], bins=20, alpha=0.7, color='purple')
        axes[1, 0].set_title('Balance Scores')
        axes[1, 0].set_xlabel('Balance Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # Energy costs
        axes[1, 1].hist(self.metrics['energy_costs'], bins=20, alpha=0.7, color='red')
        axes[1, 1].set_title('Energy Costs')
        axes[1, 1].set_xlabel('Energy Cost')
        axes[1, 1].set_ylabel('Frequency')
        
        # Success/Fail pie chart
        labels = ['Success', 'Failure']
        sizes = [results['success_rate'], 1 - results['success_rate']]
        colors = ['lightgreen', 'lightcoral']
        axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[1, 2].set_title('Success Rate')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"📊 Plots saved to {save_path}")
        
        return fig
        
    def save_results(self, results: Dict, save_path: str = "evaluation_results.json"):
        """Save evaluation results to JSON."""
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32)):
                json_results[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        self.logger.info(f"💾 Results saved to {save_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Test Donet Humanoid Walking Performance')
    parser.add_argument('checkpoint', type=str, help='Path to trained checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--episodes', type=int, default=50, help='Number of test episodes')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--output-dir', type=str, default='test_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize evaluator
        logger.info("🤖 Donet Humanoid Walking Evaluation")
        evaluator = DonetEvaluator(args.checkpoint, args.config)
        
        # Run evaluation
        results = evaluator.run_evaluation(args.episodes, args.max_steps)
        
        # Display results
        evaluator.print_results(results)
        
        # Save results and plots
        evaluator.save_results(results, output_dir / "results.json")
        evaluator.plot_results(results, output_dir / "plots.png")
        
        logger.info(f"✅ Evaluation complete! Results saved to {output_dir}")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Checkpoint file not found: {e}")
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()