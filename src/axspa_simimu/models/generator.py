import torch
import torch.nn as nn
from omegaconf import DictConfig

class ConditionalSMPLGenerator(nn.Module):
    """
    Generator for conditionally synthesizing sequences of SMPL pose parameters.
    
    Takes in a latent noise vector and a clinical condition vector, and outputs
    a sequence of 3D poses (usually in 6D rotation format).
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.condition_dim = cfg.condition_dim
        self.seq_length = cfg.seq_length
        self.representation = cfg.representation
        
        # Calculate the feature dimension for a single pose based on representation
        # SMPL typically has 24 joints (including root)
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
        
        # The input to the temporal model will be the concatenated latent and condition vectors
        input_dim = self.latent_dim + self.condition_dim
        
        # Temporal model configuration
        if arch.temporal_model.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=arch.hidden_dims[0],
                num_layers=arch.temporal_layers,
                batch_first=True,
                dropout=arch.dropout if arch.temporal_layers > 1 else 0
            )
        elif arch.temporal_model.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=arch.hidden_dims[0],
                num_layers=arch.temporal_layers,
                batch_first=True,
                dropout=arch.dropout if arch.temporal_layers > 1 else 0
            )
        else:
            raise NotImplementedError(f"Temporal model {arch.temporal_model} not implemented.")
            
        # MLP applied to each timestep output from the RNN
        layers = []
        in_dim = arch.hidden_dims[0]
        for out_dim in arch.hidden_dims[1:]:
            layers.append(nn.Linear(in_dim, out_dim))
            if arch.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            
            if arch.activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.ReLU())
                
            layers.append(nn.Dropout(arch.dropout))
            in_dim = out_dim
            
        # Final projection to the target pose representation dimension
        layers.append(nn.Linear(in_dim, self.feature_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, c):
        """
        Forward pass for the generator.
        
        Args:
            z: Latent noise vector of shape (B, latent_dim) or (B, seq_length, latent_dim)
            c: Condition vector of shape (B, condition_dim)
            
        Returns:
            out: Generated sequence of SMPL poses of shape (B, seq_length, feature_dim)
        """
        B = z.size(0)
        
        # If a single latent vector is provided, repeat it for the entire sequence
        if z.dim() == 2:
            z = z.unsqueeze(1).repeat(1, self.seq_length, 1)
            
        # Ensure condition is expanded across the sequence length
        if c.dim() == 2:
            c = c.unsqueeze(1).repeat(1, self.seq_length, 1)
            
        # Concatenate latent noise and condition features
        x = torch.cat([z, c], dim=-1)
        
        # Process through temporal model
        out, _ = self.rnn(x)  # Shape: (B, seq_length, hidden_dims[0])
        
        # Map temporal features to specific pose representation (applied per timestep)
        out = self.mlp(out)   # Shape: (B, seq_length, feature_dim)
        
        return out

