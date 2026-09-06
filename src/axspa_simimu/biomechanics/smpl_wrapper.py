"""
SMPL Kinematic tree wrapper and coordinate representations.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. to rotation matrix.
    Input: (*, 6)
    Output: (*, 3, 3)
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]

    b1 = nn.functional.normalize(a1, dim=-1)
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = nn.functional.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Extracts 6D rotation representation from 3x3 rotation matrix.
    Input: (*, 3, 3)
    Output: (*, 6)
    """
    return torch.cat([matrix[..., :3, 0], matrix[..., :3, 1]], dim=-1)


class SMPLSkeletonKinematics(nn.Module):
    """
    Hierarchical forward kinematics computation for 24-joint SMPL body model.
    """

    # SMPL 24 joint parent hierarchy
    PARENTS = [
        -1,  # 0: Pelvis
        0,   # 1: L_Hip
        0,   # 2: R_Hip
        0,   # 3: Spine1
        1,   # 4: L_Knee
        2,   # 5: R_Knee
        3,   # 6: Spine2
        4,   # 7: L_Ankle
        5,   # 8: R_Ankle
        6,   # 9: Spine3
        7,   # 10: L_Foot
        8,   # 11: R_Foot
        9,   # 12: Neck
        9,   # 13: L_Collar
        9,   # 14: R_Collar
        12,  # 15: Head
        13,  # 16: L_Shoulder
        14,  # 17: R_Shoulder
        16,  # 18: L_Elbow
        17,  # 19: R_Elbow
        18,  # 20: L_Wrist
        19,  # 21: R_Wrist
        20,  # 22: L_Hand
        21,  # 23: R_Hand
    ]

    def __init__(self, canonical_offsets: Optional[torch.Tensor] = None):
        super().__init__()
        # Default canonical bone offsets (24, 3) in meters if none provided
        if canonical_offsets is None:
            canonical_offsets = torch.zeros(24, 3)
        self.register_buffer("offsets", canonical_offsets)

    def forward_kinematics(
        self,
        rot_matrices: torch.Tensor,
        root_translation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes forward kinematics.
        Args:
            rot_matrices: (Batch, SeqLen, 24, 3, 3) local relative joint rotation matrices.
            root_translation: (Batch, SeqLen, 3) global position.

        Returns:
            global_positions: (Batch, SeqLen, 24, 3)
            global_rotations: (Batch, SeqLen, 24, 3, 3)
        """
        batch_size, seq_len, num_joints, _, _ = rot_matrices.shape
        device = rot_matrices.device

        global_rotations = [None] * num_joints
        global_positions = [None] * num_joints

        root_pos = root_translation if root_translation is not None else torch.zeros(
            batch_size, seq_len, 3, device=device
        )

        for i in range(num_joints):
            parent = self.PARENTS[i]
            if parent == -1:
                global_rotations[i] = rot_matrices[:, :, i]
                global_positions[i] = root_pos
            else:
                global_rotations[i] = torch.matmul(global_rotations[parent], rot_matrices[:, :, i])
                offset = self.offsets[i].view(1, 1, 3, 1).to(device)
                rotated_offset = torch.matmul(global_rotations[parent], offset).squeeze(-1)
                global_positions[i] = global_positions[parent] + rotated_offset

        stacked_positions = torch.stack(global_positions, dim=2)
        stacked_rotations = torch.stack(global_rotations, dim=2)

        return stacked_positions, stacked_rotations
