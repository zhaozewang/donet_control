import hydra
from omegaconf import DictConfig
import logging
import torch
from pathlib import Path

from src.utils.model_factory import create_model, create_task, create_trainer


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for REMI humanoid control training.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting REMI Humanoid Control Training")
    logger.info(f"Configuration:\n{cfg}")
    
    # Set device
    if cfg.environment.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.environment.device)
    
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(cfg.environment.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.environment.seed)
    
    # Create task
    logger.info("Creating task...")
    task = create_task(cfg.task)
    logger.info(f"Task created: {type(task).__name__}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(cfg.model, task)
    model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {type(model).__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(cfg.training, model, task)
    logger.info(f"Trainer created: {type(trainer).__name__}")
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()