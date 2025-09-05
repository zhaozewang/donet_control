#!/usr/bin/env python3
"""
Simple test script to verify the basic project structure works.
Run this to check if imports and basic model creation work before training.
"""

import torch
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    try:
        from src.models.encoding import DifferenceOfGaussiansEncoder, PerceptualEncoder
        print("✓ Encoding models imported successfully")
        
        from src.models.gru_remi import REMIGRUCell, REMIGRU
        print("✓ REMI GRU models imported successfully")
        
        from src.tasks.base_task import BaseTask
        from src.tasks.humanoid_walking import HumanoidWalkingTask
        print("✓ Task classes imported successfully")
        
        from src.training.curriculum_trainer import CurriculumTrainer
        print("✓ Training classes imported successfully")
        
        from src.utils.model_factory import create_model, create_task, create_trainer
        print("✓ Factory functions imported successfully")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
        
    return True


def test_dog_encoder():
    """Test DoG encoder functionality."""
    print("\nTesting DoG encoder...")
    
    try:
        from src.models.encoding import DifferenceOfGaussiansEncoder
        
        # Create encoder
        encoder = DifferenceOfGaussiansEncoder(
            n_joints=18,
            n_units_per_joint=32,
            sigma_excitation=0.1,
            sigma_inhibition=0.3,
            inhibition_strength=0.8
        )
        
        # Test forward pass
        batch_size = 4
        joint_angles = torch.randn(batch_size, 18) * 0.5  # Small joint angles
        
        encoded = encoder(joint_angles)
        
        expected_shape = (batch_size, 18 * 32)
        assert encoded.shape == expected_shape, f"Expected {expected_shape}, got {encoded.shape}"
        assert encoded.min() >= 0.0, "DoG encoding should be non-negative"
        
        print(f"✓ DoG encoder works: input {joint_angles.shape} -> output {encoded.shape}")
        print(f"✓ Output range: [{encoded.min():.3f}, {encoded.max():.3f}]")
        
    except Exception as e:
        print(f"✗ DoG encoder test failed: {e}")
        return False
        
    return True


def test_remi_gru():
    """Test REMI GRU model."""
    print("\nTesting REMI GRU...")
    
    try:
        from src.models.gru_remi import REMIGRU
        
        # Create model
        model = REMIGRU(
            n_joints=18,
            percept_input_dim=128,
            joint_units=64,
            percept_units=64,
            place_units=128,
            planning_units=64,
            torque_units=32,
            dog_units_per_joint=16  # Smaller for testing
        )
        
        print(f"✓ REMI GRU created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test encoding mode
        batch_size = 2
        joint_angles = torch.randn(batch_size, 18) * 0.1
        perceptual_inputs = torch.randn(batch_size, 128) * 0.1
        
        hidden = model.init_hidden(batch_size, torch.device('cpu'))
        hidden, torques = model(
            joint_angles=joint_angles,
            perceptual_inputs=perceptual_inputs,
            hidden=hidden,
            mode="encoding"
        )
        
        assert hidden.shape == (batch_size, model.hidden_size)
        assert torques.shape == (batch_size, 18)
        
        print(f"✓ Encoding mode: hidden {hidden.shape}, torques {torques.shape}")
        
        # Test planning mode
        planning_goals = torch.randn(batch_size, model.planning_units) * 0.1
        hidden, torques = model(
            planning_goals=planning_goals,
            hidden=hidden,
            mode="planning"
        )
        
        print(f"✓ Planning mode: hidden {hidden.shape}, torques {torques.shape}")
        
    except Exception as e:
        print(f"✗ REMI GRU test failed: {e}")
        return False
        
    return True


def test_task():
    """Test task creation."""
    print("\nTesting task creation...")
    
    try:
        from src.tasks.humanoid_walking import HumanoidWalkingTask
        
        # Create minimal config
        config = {
            'mujoco': {
                'xml_path': 'assets/humanoid.xml',
                'timestep': 0.01,
                'frame_skip': 5,
                'episode_length': 1000
            },
            'robot': {
                'n_joints': 18,
                'joint_names': [f'joint_{i}' for i in range(18)],
                'torque_limits': [-150.0, 150.0],
                'joint_limits': [-3.14159, 3.14159]
            },
            'sensors': {
                'imu_dim': 6,
                'contact_dim': 4,
                'joint_torque_dim': 18,
                'joint_velocity_dim': 18,
                'com_dynamics_dim': 6,
                'misc_dim': 76,
                'total_percept_dim': 128
            },
            'walking': {
                'target_velocity': 1.0,
                'balance_threshold': 0.5,
                'energy_weight': 0.01,
                'smoothness_weight': 0.1
            },
            'rewards': {
                'forward_progress': 1.0,
                'balance_maintenance': 0.5,
                'energy_efficiency': 0.01,
                'smooth_gait': 0.1,
                'alive_bonus': 0.1
            }
        }
        
        task = HumanoidWalkingTask(config)
        
        print(f"✓ Task created: {task.n_joints} joints, {task.percept_dim} perceptual dims")
        
        # Test observation
        joint_angles, perceptual_data = task.get_observation()
        assert joint_angles.shape == (18,)
        assert perceptual_data.shape == (128,)
        
        print(f"✓ Observations: joints {joint_angles.shape}, percept {perceptual_data.shape}")
        
        # Test goal generation
        random_goal = task.generate_random_goal()
        structured_goal = task.generate_structured_goal(0.5)
        
        assert random_goal.shape == (18,)
        assert structured_goal.shape == (18,)
        
        print(f"✓ Goal generation: random {random_goal.shape}, structured {structured_goal.shape}")
        
    except Exception as e:
        print(f"✗ Task test failed: {e}")
        return False
        
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("REMI Humanoid Control - Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_dog_encoder()  
    all_passed &= test_remi_gru()
    all_passed &= test_task()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! Project setup is working correctly.")
        print("You can now run: python main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        
    print("=" * 50)


if __name__ == "__main__":
    main()