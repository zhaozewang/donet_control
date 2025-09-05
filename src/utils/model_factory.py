import torch
from omegaconf import DictConfig
from typing import Any

from ..models.gru_remi import REMIGRU
from ..tasks.base_task import BaseTask
from ..tasks.humanoid_walking import HumanoidWalkingTask


def create_model(model_config: DictConfig, task: BaseTask) -> torch.nn.Module:
    """
    Factory function to create models based on configuration.
    
    Args:
        model_config: Model configuration from Hydra
        task: Task instance to get observation dimensions
        
    Returns:
        Instantiated model
    """
    # Get observation dimensions from task
    obs_dims = task.observation_dims
    
    # Update model config with task-specific dimensions
    model_params = dict(model_config)
    model_params['n_joints'] = obs_dims['n_joints']
    model_params['percept_input_dim'] = obs_dims['percept_dim']
    
    # Remove the _target_ key if present
    if '_target_' in model_params:
        del model_params['_target_']
    
    # Create REMI GRU model
    if model_config._target_ == 'src.models.gru_remi.REMIGRU':
        model = REMIGRU(**model_params)
    else:
        raise ValueError(f"Unknown model target: {model_config._target_}")
    
    return model


def create_task(task_config: DictConfig) -> BaseTask:
    """
    Factory function to create tasks based on configuration.
    
    Args:
        task_config: Task configuration from Hydra
        
    Returns:
        Instantiated task
    """
    # Remove the _target_ key for parameter passing
    task_params = dict(task_config)
    if '_target_' in task_params:
        target = task_params.pop('_target_')
    else:
        target = 'src.tasks.humanoid_walking.HumanoidWalkingTask'
    
    # Create task instance
    if target == 'src.tasks.humanoid_walking.HumanoidWalkingTask':
        task = HumanoidWalkingTask(task_params)
    else:
        raise ValueError(f"Unknown task target: {target}")
    
    return task


def create_trainer(
    trainer_config: DictConfig,
    model: torch.nn.Module,
    task: BaseTask
) -> Any:
    """
    Factory function to create trainers based on configuration.
    
    Args:
        trainer_config: Training configuration from Hydra
        model: Model instance
        task: Task instance
        
    Returns:
        Instantiated trainer
    """
    from ..training.curriculum_trainer import CurriculumTrainer
    
    # Remove the _target_ key for parameter passing
    if hasattr(trainer_config, '_target_'):
        target = trainer_config._target_
    else:
        target = 'src.training.curriculum_trainer.CurriculumTrainer'
    
    # Create trainer instance
    if target == 'src.training.curriculum_trainer.CurriculumTrainer':
        trainer = CurriculumTrainer(model, task, trainer_config)
    else:
        raise ValueError(f"Unknown trainer target: {target}")
    
    return trainer