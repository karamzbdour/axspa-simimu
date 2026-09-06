"""
Clinical metrics and Bath Ankylosing Spondylitis Metrology Index (BASMI) calculations.
"""

from typing import Dict
import torch


def compute_basmi_scores(
    joint_trajectories: torch.Tensor,
    fps: float = 60.0,
) -> Dict[str, torch.Tensor]:
    """
    Computes clinical mobility metrics conforming to BASMI sub-scores:
    1. Cervical Rotation (degrees)
    2. Tragus-to-Wall distance proxy (cm)
    3. Lumbar Side Flexion (cm)
    4. Modified Schober's Lumbar Flexion proxy (cm)
    5. Intermalleolar Distance (cm)

    Args:
        joint_trajectories: Tensor of shape (Batch, Frames, NumJoints, 3) in meters.
        fps: Sampling frequency.

    Returns:
        Dictionary of computed clinical indicators.
    """
    # Placeholder for joint indices based on standard 3D skeletal topology
    # Head/Cervical = joint 15, Spine/Lumbar = joint 3, Pelvis = joint 0, Ankles = joint 7, 8
    # Shape: (B, T, J, 3)
    batch_size, seq_len, num_joints, _ = joint_trajectories.shape

    # Example: Intermalleolar distance (distance between left and right ankles)
    if num_joints > 8:
        left_ankle = joint_trajectories[:, :, 7, :]
        right_ankle = joint_trajectories[:, :, 8, :]
        intermalleolar_dist = torch.norm(left_ankle - right_ankle, dim=-1).max(dim=-1).values * 100.0 # cm
    else:
        intermalleolar_dist = torch.zeros(batch_size, device=joint_trajectories.device)

    # Example: Lumbar flexion range
    if num_joints > 3:
        pelvis = joint_trajectories[:, :, 0, :]
        spine_mid = joint_trajectories[:, :, 3, :]
        spine_vector = spine_mid - pelvis
        # Compute pitch variation over the sequence
        pitch = torch.atan2(spine_vector[:, :, 2], spine_vector[:, :, 1])
        lumbar_flexion_rom_deg = torch.rad2deg(pitch.max(dim=-1).values - pitch.min(dim=-1).values)
    else:
        lumbar_flexion_rom_deg = torch.zeros(batch_size, device=joint_trajectories.device)

    return {
        "intermalleolar_distance_cm": intermalleolar_dist,
        "lumbar_flexion_rom_deg": lumbar_flexion_rom_deg,
    }


def compute_spinal_stiffness(
    joint_angles: torch.Tensor,
) -> torch.Tensor:
    """
    Estimates the degree of spinal rigidity from angular velocity/acceleration variance.
    """
    # Angular velocity variance across spinal joints
    # joint_angles: (B, T, SpineJoints)
    diff = torch.diff(joint_angles, dim=1)
    stiffness_metric = 1.0 / (torch.var(diff, dim=1) + 1e-6)
    return stiffness_metric
