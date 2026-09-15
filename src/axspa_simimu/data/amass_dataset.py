import os
import glob
import hydra
import torch
import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig

# Assuming these exist in your smpl_wrapper as seen earlier
from axspa_simimu.biomechanics.smpl_wrapper import matrix_to_rotation_6d

def axis_angle_to_matrix(a: torch.Tensor) -> torch.Tensor:
    """
    Vectorized conversion from axis-angle to 3x3 rotation matrix
    using Rodrigues' rotation formula.
    """
    angle = torch.norm(a, dim=-1, keepdim=True)
    axis = a / (angle + 1e-8)
    
    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)
    
    rx, ry, rz = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(rx)
    
    K = torch.stack([
        zeros, -rz, ry,
        rz, zeros, -rx,
        -ry, rx, zeros
    ], dim=-1).view(*rx.shape, 3, 3)
    
    I = torch.eye(3, device=a.device, dtype=a.dtype).expand(*rx.shape, 3, 3)
    
    R = cos * I + (1 - cos) * torch.matmul(axis.unsqueeze(-1), axis.unsqueeze(-2)) + sin * K
    
    # Handle angle=0 smoothly
    R = torch.where((angle < 1e-6).unsqueeze(-1), I, R)
    return R


class AMASSDataset(Dataset):
    """
    PyTorch Dataset for ingesting local AMASS .npz files.
    Precomputes framing, subsampling, and representation conversion once during init.
    """
    def __init__(self, cfg: DictConfig):
        # Resolve data_dir robustly across execution working directories
        raw_data_dir = cfg.dataset.data_dir
        if os.path.isabs(raw_data_dir):
            self.data_dir = raw_data_dir
        elif os.path.exists(raw_data_dir):
            self.data_dir = os.path.abspath(raw_data_dir)
        else:
            # Fallback to repo root if run from a subfolder
            try:
                self.data_dir = hydra.utils.to_absolute_path(raw_data_dir)
            except Exception:
                self.data_dir = os.path.abspath(raw_data_dir)
                
        self.seq_length = cfg.dataset.seq_length
        self.stride = cfg.dataset.stride
        self.target_fps = cfg.dataset.sampling_rate_hz
        
        self.representation = cfg.generator.representation if "generator" in cfg else cfg.model.representation
        self.condition_dim = cfg.generator.condition_dim if "generator" in cfg else cfg.model.condition_dim
        
        # 1. Find all .npz files in the target directory
        npz_files = glob.glob(os.path.join(self.data_dir, "**/*.npz"), recursive=True)
        print(f"Found {len(npz_files)} .npz files in {self.data_dir}")
        
        raw_windows = []
        
        # 2. Extract and slice sequences into memory
        for file_path in npz_files:
            try:
                data = np.load(file_path)
                poses = data["poses"]  # Shape: (N, 156) for SMPL+H
                mocap_fps = float(data.get("mocap_framerate", 120.0))
                
                # SLICE: First 72 dims for the 24 standard SMPL joints
                poses = poses[:, :72]
                
                # SUBSAMPLE: Adjust framerate (e.g., 120Hz -> 60Hz)
                if mocap_fps > self.target_fps:
                    step = max(1, int(np.round(mocap_fps / self.target_fps)))
                    poses = poses[::step]
                
                N = poses.shape[0]
                if N < self.seq_length:
                    continue
                    
                # SLIDING WINDOW: Cut into overlapping chunks
                for start_idx in range(0, N - self.seq_length + 1, self.stride):
                    window = poses[start_idx : start_idx + self.seq_length]
                    raw_windows.append(window)
                    
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")
                
        # 3. Vectorize and precompute representation once
        if len(raw_windows) > 0:
            all_poses = torch.tensor(np.array(raw_windows), dtype=torch.float32)  # (N, Seq, 72)
            del raw_windows  # Free temporary array memory
            
            if self.representation == "rotation_6d":
                N_samples = all_poses.shape[0]
                poses_24 = all_poses.view(-1, 24, 3)
                rot_matrix = axis_angle_to_matrix(poses_24)
                rot_6d = matrix_to_rotation_6d(rot_matrix)
                self.tensors = rot_6d.view(N_samples, self.seq_length, 144).contiguous()
            else:
                self.tensors = all_poses.contiguous()
        else:
            feat_dim = 144 if self.representation == "rotation_6d" else 72
            self.tensors = torch.empty(0, self.seq_length, feat_dim, dtype=torch.float32)
            
        self.condition = torch.zeros(self.condition_dim, dtype=torch.float32)
        print(f"Dataset initialized with {len(self.tensors)} valid motion windows (Shape: {list(self.tensors.shape)}).")
        
    def __len__(self):
        return len(self.tensors)
        
    def __getitem__(self, idx):
        return self.tensors[idx], self.condition

