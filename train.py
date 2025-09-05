#!/usr/bin/env python3
"""
Training script for Donet Humanoid Control System.

This is the main training entry point that runs the full curriculum:
1. Phase 1: Pure exploration with high noise
2. Phase 2: Noise reduction with goal-directed training  
3. Phase 3: Refined control with minimal noise
4. Phase 4: PPO integration for high-level walking strategy
"""

import hydra
from omegaconf import DictConfig
import logging
import torch
from pathlib import Path

from src.utils.model_factory import create_model, create_task, create_trainer


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training entry point.
    
    Usage:
        python train.py                                    # Default training
        python train.py rl.use_ppo=false                  # Donet-only training
        python train.py model.joint_units=512             # Custom model size
        python train.py training.phases.phase_1.duration_iterations=20000  # Custom curriculum
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🤖 Starting Donet Humanoid Control Training")
    logger.info("=" * 60)
    
    # Set device
    if cfg.environment.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.environment.device)
    
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {cfg.environment.seed}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.environment.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.environment.seed)
        torch.cuda.manual_seed_all(cfg.environment.seed)
    
    # Create task
    logger.info("Creating humanoid walking task...")
    task = create_task(cfg.task)
    logger.info(f"✓ Task: {type(task).__name__}")
    logger.info(f"  - Joints: {task.n_joints}")
    logger.info(f"  - Sensors: {task.percept_dim} dimensions")
    
    # Create model
    logger.info("Creating Donet model...")
    model = create_model(cfg.model, task)
    model.to(device)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"✓ Model: {type(model).__name__}")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Hidden size: {model.hidden_size}")
    logger.info(f"  - Unit groups: J={model.joint_units}, P={model.percept_units}, "
                f"L={model.place_units}, Plan={model.planning_units}, T={model.torque_units}")
    
    # Create trainer
    logger.info("Creating curriculum trainer...")
    trainer = create_trainer(cfg, model, task)
    logger.info(f"✓ Trainer: {type(trainer).__name__}")
    
    # Display training configuration
    logger.info("Training Configuration:")
    logger.info(f"  - Max iterations: {cfg.global.max_iterations:,}")
    logger.info(f"  - Batch size: {cfg.global.batch_size}")
    logger.info(f"  - Learning rate: {cfg.global.learning_rate}")
    
    # PPO configuration
    if cfg.rl.use_ppo:
        logger.info(f"  - PPO enabled: starts at iteration {cfg.rl.ppo_start_iteration:,}")
    else:
        logger.info("  - PPO disabled: Donet-only training")
    
    # Phase information
    logger.info("Curriculum Phases:")
    for i, phase_name in enumerate(['phase_1', 'phase_2', 'phase_3'], 1):
        phase = cfg.training.phases[phase_name]
        logger.info(f"  Phase {i}: {phase.name} ({phase.duration_iterations:,} iterations)")
    
    logger.info("=" * 60)
    logger.info("🚀 Starting training...")
    
    # Start training
    try:
        trainer.train()
        logger.info("🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
        logger.info("Saving checkpoint...")
        trainer._save_checkpoint()
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.exception("Full traceback:")
        raise
    
    finally:
        logger.info("Training session ended")


if __name__ == "__main__":
    main()