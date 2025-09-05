import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import numpy as np


class DonetGRUCell(nn.Module):
    """
    Modified GRU cell for Donet architecture with specialized unit groups.
    
    State vector is partitioned into functional groups:
    - Joint units: Process DoG-encoded joint angles  
    - Perceptual units: Process MuJoCo sensory data
    - Place units: Associate configurations with contexts
    - Planning units: Generate action sequences for internal simulation
    - Torque units: Generate joint torque commands
    """
    
    def __init__(
        self,
        input_size: int,
        joint_units: int,
        percept_units: int,
        place_units: int,
        planning_units: int,
        torque_units: int,
        bias: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.joint_units = joint_units
        self.percept_units = percept_units
        self.place_units = place_units
        self.planning_units = planning_units
        self.torque_units = torque_units
        
        self.hidden_size = (
            joint_units + percept_units + place_units + 
            planning_units + torque_units
        )
        
        # Define unit group indices
        self.joint_idx = (0, joint_units)
        self.percept_idx = (joint_units, joint_units + percept_units)
        self.place_idx = (
            joint_units + percept_units,
            joint_units + percept_units + place_units
        )
        self.planning_idx = (
            joint_units + percept_units + place_units,
            joint_units + percept_units + place_units + planning_units
        )
        self.torque_idx = (
            joint_units + percept_units + place_units + planning_units,
            self.hidden_size
        )
        
        # Standard GRU gates
        self.weight_ih_reset = nn.Linear(input_size, self.hidden_size, bias=bias)
        self.weight_hh_reset = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.weight_ih_update = nn.Linear(input_size, self.hidden_size, bias=bias)
        self.weight_hh_update = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.weight_ih_new = nn.Linear(input_size, self.hidden_size, bias=bias)
        self.weight_hh_new = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        
        self._init_identity_projection()
    
    def _init_identity_projection(self):
        """Initialize input projection as identity and freeze it."""
        # Make input layers identity projection with frozen weights
        for layer in [self.weight_ih_reset, self.weight_ih_update, self.weight_ih_new]:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False if layer.bias is not None else False
            
            # Initialize as identity where possible
            if self.input_size <= self.hidden_size:
                nn.init.zeros_(layer.weight)
                layer.weight[:self.input_size, :self.input_size] = torch.eye(self.input_size)
            else:
                nn.init.zeros_(layer.weight)
                layer.weight[:, :self.input_size] = torch.eye(self.hidden_size, self.input_size)
            
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def forward(
        self, 
        input: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None,
        mode: str = "encoding"
    ) -> torch.Tensor:
        """
        Forward pass with mode-specific behavior.
        
        Args:
            input: [batch_size, input_size] input tensor (padded with zeros for unused parts)
            hidden: [batch_size, hidden_size] previous hidden state
            mode: "encoding" (encoder receives input, planning gets zeros) or 
                  "planning" (encoder gets zeros, planning receives goal)
                  
        Returns:
            new_hidden: [batch_size, hidden_size] updated hidden state
        """
        if hidden is None:
            hidden = torch.zeros(
                input.size(0), self.hidden_size,
                dtype=input.dtype, device=input.device
            )
        
        # Apply input masking based on mode
        if mode == "encoding":
            # During encoding: zero out planning input regions
            masked_input = input.clone()
            if input.size(1) > self.joint_units + self.percept_units:
                masked_input[:, self.joint_units + self.percept_units:] = 0.0
        elif mode == "planning":
            # During planning: zero out encoder input regions  
            masked_input = torch.zeros_like(input)
            if input.size(1) > self.joint_units + self.percept_units:
                masked_input[:, self.joint_units + self.percept_units:] = input[:, self.joint_units + self.percept_units:]
        else:
            masked_input = input
        
        # Standard GRU computation with masked input
        gi = self.weight_ih_reset(masked_input)
        gh = self.weight_hh_reset(hidden)
        reset_gate = torch.sigmoid(gi + gh)
        
        gi = self.weight_ih_update(masked_input)
        gh = self.weight_hh_update(hidden)
        update_gate = torch.sigmoid(gi + gh)
        
        gi = self.weight_ih_new(masked_input)
        gh = self.weight_hh_new(reset_gate * hidden)
        new_gate = torch.tanh(gi + gh)
        
        new_hidden = (1 - update_gate) * hidden + update_gate * new_gate
        
        return new_hidden
    
    def get_unit_groups(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract unit groups from hidden state."""
        return {
            'joint': hidden[:, self.joint_idx[0]:self.joint_idx[1]],
            'percept': hidden[:, self.percept_idx[0]:self.percept_idx[1]], 
            'place': hidden[:, self.place_idx[0]:self.place_idx[1]],
            'planning': hidden[:, self.planning_idx[0]:self.planning_idx[1]],
            'torque': hidden[:, self.torque_idx[0]:self.torque_idx[1]]
        }


class DonetGRU(nn.Module):
    """
    Full Donet GRU network for humanoid control.
    
    Combines DoG joint encoding, perceptual processing, and specialized GRU cell
    with planning capabilities.
    """
    
    def __init__(
        self,
        n_joints: int,
        percept_input_dim: int,
        joint_units: int,
        percept_units: int, 
        place_units: int,
        planning_units: int,
        torque_units: int,
        dog_units_per_joint: int = 32,
        dog_sigma_excite: float = 0.1,
        dog_sigma_inhibit: float = 0.3,
        dog_inhibit_strength: float = 0.8,
        joint_range: Tuple[float, float] = (-np.pi, np.pi)
    ):
        super().__init__()
        
        self.n_joints = n_joints
        self.percept_input_dim = percept_input_dim
        
        # DoG encoder for joint angles
        from .encoding import DifferenceOfGaussiansEncoder, PerceptualEncoder
        
        self.dog_encoder = DifferenceOfGaussiansEncoder(
            n_joints=n_joints,
            n_units_per_joint=dog_units_per_joint,
            sigma_excitation=dog_sigma_excite,
            sigma_inhibition=dog_sigma_inhibit,
            inhibition_strength=dog_inhibit_strength,
            joint_range=joint_range
        )
        
        # Perceptual encoder
        self.percept_encoder = PerceptualEncoder(
            input_dim=percept_input_dim,
            encoding_dim=percept_units,
            hidden_dim=min(256, percept_units * 2)
        )
        
        # Total input size to GRU
        dog_encoding_dim = n_joints * dog_units_per_joint
        self.input_size = dog_encoding_dim + percept_units + planning_units
        
        # Donet GRU cell
        self.gru_cell = DonetGRUCell(
            input_size=self.input_size,
            joint_units=joint_units,
            percept_units=percept_units,
            place_units=place_units,
            planning_units=planning_units,
            torque_units=torque_units
        )
        
        # Output projection for torque commands
        self.torque_projection = nn.Linear(torque_units, n_joints)
        
        # Store dimensions for easy access
        self.hidden_size = self.gru_cell.hidden_size
        self.joint_units = joint_units
        self.percept_units = percept_units
        self.place_units = place_units
        self.planning_units = planning_units
        self.torque_units = torque_units
    
    def encode_inputs(
        self, 
        joint_angles: torch.Tensor,
        perceptual_inputs: torch.Tensor
    ) -> torch.Tensor:
        """Encode joint angles and perceptual inputs."""
        dog_encoded = self.dog_encoder(joint_angles)
        percept_encoded = self.percept_encoder(perceptual_inputs)
        
        # Concatenate encodings with zero-padded planning region
        batch_size = joint_angles.shape[0]
        planning_zeros = torch.zeros(
            batch_size, self.planning_units,
            dtype=joint_angles.dtype, device=joint_angles.device
        )
        
        return torch.cat([dog_encoded, percept_encoded, planning_zeros], dim=1)
    
    def forward(
        self,
        joint_angles: Optional[torch.Tensor] = None,
        perceptual_inputs: Optional[torch.Tensor] = None,
        planning_goals: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        mode: str = "encoding"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with mode-specific input handling.
        
        Args:
            joint_angles: [batch, n_joints] joint angles for encoding mode
            perceptual_inputs: [batch, percept_dim] perceptual data for encoding mode  
            planning_goals: [batch, planning_units] goal states for planning mode
            hidden: [batch, hidden_size] previous hidden state
            mode: "encoding" or "planning"
            
        Returns:
            hidden: [batch, hidden_size] new hidden state
            torque_commands: [batch, n_joints] torque outputs
        """
        batch_size = (joint_angles.shape[0] if joint_angles is not None else
                     planning_goals.shape[0] if planning_goals is not None else 1)
        
        if mode == "encoding":
            if joint_angles is None or perceptual_inputs is None:
                raise ValueError("joint_angles and perceptual_inputs required for encoding mode")
            input_tensor = self.encode_inputs(joint_angles, perceptual_inputs)
            
        elif mode == "planning":
            if planning_goals is None:
                raise ValueError("planning_goals required for planning mode")
            # Zero-pad encoder regions, use planning goals
            dog_zeros = torch.zeros(
                batch_size, self.n_joints * self.dog_encoder.n_units_per_joint,
                dtype=planning_goals.dtype, device=planning_goals.device
            )
            percept_zeros = torch.zeros(
                batch_size, self.percept_units,
                dtype=planning_goals.dtype, device=planning_goals.device  
            )
            input_tensor = torch.cat([dog_zeros, percept_zeros, planning_goals], dim=1)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # GRU forward pass
        hidden = self.gru_cell(input_tensor, hidden, mode=mode)
        
        # Extract torque commands
        unit_groups = self.gru_cell.get_unit_groups(hidden)
        torque_commands = self.torque_projection(unit_groups['torque'])
        
        return hidden, torque_commands
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(
            batch_size, self.hidden_size,
            dtype=torch.float32, device=device
        )