import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from omegaconf import DictConfig

class KinematicDiscriminator(nn.Module):
    """
    Discriminator for validating generated kinematic sequences.
    
    Uses a 1D Convolutional temporal architecture to discriminate between real 
    and generated (fake) sequences of SMPL pose representations, conditioned on clinical severity.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.condition_dim = cfg.condition_dim
        self.seq_length = cfg.seq_length
        self.representation = cfg.representation
        
        # Feature dimension is based on 24 SMPL joints
        self.num_joints = 24
        if self.representation == "rotation_6d":
            self.feature_dim = self.num_joints * 6
        elif self.representation == "quaternion":
            self.feature_dim = self.num_joints * 4
        elif self.representation == "axis_angle":
            self.feature_dim = self.num_joints * 3
        else:
            raise ValueError(f"Unknown representation: {self.representation}")
            
        arch = cfg.architecture
        
        # Concatenate the pose feature dimension and condition dimension for 1D conv inputs
        in_channels = self.feature_dim + self.condition_dim
        
        layers = []
        for out_channels in arch.conv_channels:
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=arch.kernel_size,
                stride=arch.stride,
                padding=arch.kernel_size // 2
            )
            
            # WGAN-GP often benefits from (or can optionally use) spectral normalization
            if arch.use_spectral_norm:
                conv = spectral_norm(conv)
                
            layers.append(conv)
            
            if arch.activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU(arch.negative_slope))
            else:
                layers.append(nn.ReLU())
                
            layers.append(nn.Dropout(arch.dropout))
            
            in_channels = out_channels
            
        self.conv_net = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        
        # Calculate the size of the flattened output dynamically
        dummy_input = torch.zeros(1, self.feature_dim + self.condition_dim, self.seq_length)
        with torch.no_grad():
            dummy_out = self.conv_net(dummy_input)
            flatten_dim = dummy_out.view(1, -1).shape[1]
            
        # Final linear layer to output a scalar score
        final_layer = nn.Linear(flatten_dim, 1)
        if arch.use_spectral_norm:
            final_layer = spectral_norm(final_layer)
            
        self.fc = final_layer

    def forward(self, x, c):
        """
        Forward pass for the discriminator.
        
        Args:
            x: Sequence of SMPL poses, real or fake, of shape (B, seq_length, feature_dim)
            c: Condition vector of shape (B, condition_dim)
            
        Returns:
            out: Discriminator score of shape (B, 1)
        """
        B = x.size(0)
        
        # Expand condition over the temporal dimension
        if c.dim() == 2:
            c = c.unsqueeze(1).repeat(1, self.seq_length, 1)
            
        # Concatenate the pose data with the condition
        x_c = torch.cat([x, c], dim=-1)  # Shape: (B, seq_length, feature_dim + condition_dim)
        
        # PyTorch Conv1d expects shape (B, Channels, Length)
        x_c = x_c.transpose(1, 2)
        
        out = self.conv_net(x_c)
        out = self.flatten(out)
        out = self.fc(out)
        
        return out

