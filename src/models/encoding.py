import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class DifferenceOfGaussiansEncoder(nn.Module):
    """
    Difference of Gaussians (DoG) encoding for joint angles.
    
    Creates sparse encoding with narrow excitation and broader inhibition,
    providing lateral inhibition between neighboring units.
    """
    
    def __init__(
        self,
        n_joints: int,
        n_units_per_joint: int,
        sigma_excitation: float = 0.1,
        sigma_inhibition: float = 0.3,
        inhibition_strength: float = 0.8,
        joint_range: Tuple[float, float] = (0.0, 2 * np.pi)
    ):
        """
        Args:
            n_joints: Number of joints to encode
            n_units_per_joint: Number of encoding units per joint
            sigma_excitation: Standard deviation for excitatory component
            sigma_inhibition: Standard deviation for inhibitory component
            inhibition_strength: Strength of inhibition (k parameter)
            joint_range: Range of joint angles (min, max)
        """
        super().__init__()
        
        self.n_joints = n_joints
        self.n_units_per_joint = n_units_per_joint
        self.sigma_excitation = sigma_excitation
        self.sigma_inhibition = sigma_inhibition
        self.inhibition_strength = inhibition_strength
        self.joint_min, self.joint_max = joint_range
        
        # Create preferred angles evenly distributed across joint range
        preferred_angles = torch.linspace(
            self.joint_min, self.joint_max, n_units_per_joint
        )
        # Repeat for all joints: [joint0_unit0, joint0_unit1, ..., joint1_unit0, ...]
        self.register_buffer(
            'preferred_angles',
            preferred_angles.repeat(n_joints)
        )
        
        self.total_units = n_joints * n_units_per_joint
    
    def forward(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Encode joint angles using DoG tuning curves.
        
        Args:
            joint_angles: [batch_size, n_joints] joint angles in radians
            
        Returns:
            encoded: [batch_size, n_joints * n_units_per_joint] sparse encoding
        """
        batch_size = joint_angles.shape[0]
        
        # Repeat joint angles for all units: [batch, joint0, joint0, ..., joint1, joint1, ...]
        angles_expanded = joint_angles.repeat_interleave(self.n_units_per_joint, dim=1)
        
        # Compute differences from preferred angles
        angle_diffs = angles_expanded - self.preferred_angles.unsqueeze(0)
        
        # Excitatory component
        excitation = torch.exp(
            -0.5 * (angle_diffs / self.sigma_excitation) ** 2
        )
        
        # Inhibitory component
        inhibition = torch.exp(
            -0.5 * (angle_diffs / self.sigma_inhibition) ** 2
        )
        
        # DoG response
        response = excitation - self.inhibition_strength * inhibition
        
        # Clamp to ensure non-negative sparse activation
        response = torch.clamp(response, min=0.0)
        
        return response


class PerceptualEncoder(nn.Module):
    """
    Encoder for MuJoCo perceptual inputs including IMU, forces, velocities etc.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2
    ):
        """
        Args:
            input_dim: Dimension of raw perceptual inputs
            encoding_dim: Dimension of encoded perceptual features
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, encoding_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, perceptual_inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode perceptual inputs.
        
        Args:
            perceptual_inputs: [batch_size, input_dim] raw perceptual data
            
        Returns:
            encoded: [batch_size, encoding_dim] encoded perceptual features
        """
        return self.encoder(perceptual_inputs)