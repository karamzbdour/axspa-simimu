"""
Deterministic forward kinematic sensor synthesis (linear acceleration and angular velocity).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class VirtualIMUSimulator(nn.Module):
    """
    Synthesizes local-frame tri-axial acceleration a(t) and angular velocity omega(t)
    from global 3D joint/vertex positions and rotation matrices.
    """

    def __init__(self, fps: float = 60.0, gravity: float = 9.81):
        super().__init__()
        self.fps = fps
        self.dt = 1.0 / fps
        self.gravity_const = gravity
        # Standard gravity vector in global frame (upwards z or y depending on world frame convention)
        self.register_buffer("gravity_vec", torch.tensor([0.0, 0.0, gravity]))

    def compute_acceleration(
        self,
        positions: torch.Tensor,
        global_rotations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes local sensor-frame linear acceleration including gravity:
        a_local(t) = R(t)^T * ( d^2 x / dt^2 + g )

        Args:
            positions: (Batch, SeqLen, NumSensors, 3) in meters.
            global_rotations: (Batch, SeqLen, NumSensors, 3, 3) rotation matrices.

        Returns:
            local_acc: (Batch, SeqLen, NumSensors, 3) in m/s^2.
        """
        batch_size, seq_len, num_sensors, _ = positions.shape
        dt = self.dt

        # Second-order central numerical differentiation for internal steps
        # Forward diff for first step, backward for last step
        vel = torch.zeros_like(positions)
        vel[:, 1:-1] = (positions[:, 2:] - positions[:, :-2]) / (2.0 * dt)
        vel[:, 0] = (positions[:, 1] - positions[:, 0]) / dt
        vel[:, -1] = (positions[:, -1] - positions[:, -2]) / dt

        acc_global = torch.zeros_like(positions)
        acc_global[:, 1:-1] = (vel[:, 2:] - vel[:, :-2]) / (2.0 * dt)
        acc_global[:, 0] = (vel[:, 1] - vel[:, 0]) / dt
        acc_global[:, -1] = (vel[:, -1] - vel[:, -2]) / dt

        # Add gravitational acceleration
        acc_with_gravity = acc_global + self.gravity_vec.view(1, 1, 1, 3)

        # Transform to local sensor frame: a_local = R^T * a_global
        rot_t = global_rotations.transpose(-1, -2) # (B, T, S, 3, 3)
        acc_local = torch.matmul(rot_t, acc_with_gravity.unsqueeze(-1)).squeeze(-1)

        return acc_local

    def compute_angular_velocity(
        self,
        global_rotations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes local-frame angular velocity omega(t) from rotation matrices:
        R_dot * R^T = [omega]x

        Args:
            global_rotations: (Batch, SeqLen, NumSensors, 3, 3)

        Returns:
            local_omega: (Batch, SeqLen, NumSensors, 3) in rad/s.
        """
        dt = self.dt
        # Relative rotation between successive frames: R_rel = R_{t}^T * R_{t+1}
        r_t = global_rotations[:, :-1]
        r_next = global_rotations[:, 1:]
        r_rel = torch.matmul(r_t.transpose(-1, -2), r_next) # (B, T-1, S, 3, 3)

        # Extract axis-angle / angular velocity from rotation matrix
        # tr(R) = 1 + 2*cos(theta)
        trace = r_rel[..., 0, 0] + r_rel[..., 1, 1] + r_rel[..., 2, 2]
        cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta) # angle

        # Unnormalized skew components
        w_x = r_rel[..., 2, 1] - r_rel[..., 1, 2]
        w_y = r_rel[..., 0, 2] - r_rel[..., 2, 0]
        w_z = r_rel[..., 1, 0] - r_rel[..., 0, 1]
        w_skew = torch.stack([w_x, w_y, w_z], dim=-1)

        sin_theta = torch.sin(theta).unsqueeze(-1)
        scale = torch.where(
            theta.unsqueeze(-1) > 1e-5,
            (theta.unsqueeze(-1) / (2.0 * sin_theta + 1e-7)) / dt,
            torch.ones_like(w_skew) / dt,
        )
        omega = w_skew * scale

        # Pad last frame to keep sequence length equal
        omega_full = torch.cat([omega, omega[:, -1:]], dim=1)
        return omega_full

    def forward(
        self,
        positions: torch.Tensor,
        global_rotations: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        acc = self.compute_acceleration(positions, global_rotations)
        gyro = self.compute_angular_velocity(global_rotations)
        return {"accel": acc, "gyro": gyro, "rotations": global_rotations}
