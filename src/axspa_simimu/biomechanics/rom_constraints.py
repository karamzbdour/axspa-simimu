"""
Range of Motion (ROM) constraints and clinical limit validation for axSpA.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class ROMConstraintManager:
    """
    Manages anatomical and pathological Range of Motion (ROM) boundaries
    for healthy controls and axial spondyloarthritis (axSpA) severity conditions.
    """

    # Anatomical baseline limits in degrees [min_angle, max_angle]
    DEFAULT_HEALTHY_ROM_DEG = {
        "lumbar_flexion": (0.0, 75.0),
        "lumbar_extension": (0.0, 30.0),
        "lumbar_lateral_flexion": (-35.0, 35.0),
        "cervical_flexion": (0.0, 60.0),
        "cervical_extension": (0.0, 70.0),
        "cervical_rotation": (-80.0, 80.0),
        "thoracic_rotation": (-35.0, 35.0),
    }

    def __init__(self, custom_limits: Optional[Dict[str, Tuple[float, float]]] = None):
        self.limits_deg = custom_limits or self.DEFAULT_HEALTHY_ROM_DEG
        self.limits_rad = {
            k: (np.radians(v[0]), np.radians(v[1])) for k, v in self.limits_deg.items()
        }

    def compute_pathological_bounds(
        self, severity_vector: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Scale ROM limits based on clinical severity vector alpha in [0, 1]^3
        alpha = [cervical_stiffness, thoracic_stiffness, lumbar_stiffness]
        """
        # alpha shape: (batch_size, 3) or (3,)
        if severity_vector.dim() == 1:
            severity_vector = severity_vector.unsqueeze(0)

        c_stiff = severity_vector[:, 0:1]
        t_stiff = severity_vector[:, 1:2]
        l_stiff = severity_vector[:, 2:3]

        # Minimum residual joint angle factor under complete ankylosis
        residual_factor = 0.15

        scaled_bounds = {}
        # Lumbar joints scaled by lumbar stiffness
        for k in ["lumbar_flexion", "lumbar_extension", "lumbar_lateral_flexion"]:
            min_val, max_val = self.limits_rad[k]
            scale = 1.0 - l_stiff * (1.0 - residual_factor)
            scaled_bounds[k] = (min_val * scale, max_val * scale)

        # Cervical joints scaled by cervical stiffness
        for k in ["cervical_flexion", "cervical_extension", "cervical_rotation"]:
            min_val, max_val = self.limits_rad[k]
            scale = 1.0 - c_stiff * (1.0 - residual_factor)
            scaled_bounds[k] = (min_val * scale, max_val * scale)

        # Thoracic joints scaled by thoracic stiffness
        for k in ["thoracic_rotation"]:
            min_val, max_val = self.limits_rad[k]
            scale = 1.0 - t_stiff * (1.0 - residual_factor)
            scaled_bounds[k] = (min_val * scale, max_val * scale)

        return scaled_bounds
