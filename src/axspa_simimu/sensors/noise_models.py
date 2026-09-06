"""
Realistic IMU noise models: Gaussian white noise, bias random walk drift, and soft-tissue artifacts.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn


class IMUNoiseSimulator(nn.Module):
    """
    Applies synthetic sensor noise, random-walk bias drift, mounting misalignment,
    and soft-tissue artifact (STA) wobbling mass perturbations.
    """

    def __init__(
        self,
        accel_noise_std: float = 0.15,
        gyro_noise_std: float = 0.03,
        accel_drift_walk: float = 0.002,
        gyro_drift_walk: float = 0.0005,
        simulate_soft_tissue_artifact: bool = True,
        sta_frequency_hz: float = 3.5,
        sta_amplitude_m: float = 0.008,
        fps: float = 60.0,
    ):
        super().__init__()
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_drift_walk = accel_drift_walk
        self.gyro_drift_walk = gyro_drift_walk
        self.simulate_sta = simulate_soft_tissue_artifact
        self.sta_freq = sta_frequency_hz
        self.sta_amp = sta_amplitude_m
        self.fps = fps

    def forward(
        self,
        clean_imu: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            clean_imu: Dict with 'accel' (B, T, S, 3) and 'gyro' (B, T, S, 3)

        Returns:
            noisy_imu: Dict with perturbed 'accel' and 'gyro'
        """
        accel = clean_imu["accel"]
        gyro = clean_imu["gyro"]
        batch_size, seq_len, num_sensors, _ = accel.shape
        device = accel.device

        # 1. Add Gaussian White Noise
        if self.accel_noise_std > 0:
            acc_noise = torch.randn_like(accel) * self.accel_noise_std
            accel = accel + acc_noise

        if self.gyro_noise_std > 0:
            gyro_noise = torch.randn_like(gyro) * self.gyro_noise_std
            gyro = gyro + gyro_noise

        # 2. Bias Random Walk Drift
        if self.accel_drift_walk > 0:
            acc_steps = torch.randn_like(accel) * self.accel_drift_walk
            acc_drift = torch.cumsum(acc_steps, dim=1)
            accel = accel + acc_drift

        if self.gyro_drift_walk > 0:
            gyro_steps = torch.randn_like(gyro) * self.gyro_drift_walk
            gyro_drift = torch.cumsum(gyro_steps, dim=1)
            gyro = gyro + gyro_drift

        # 3. Soft-tissue Artifact (STA) harmonic vibration
        if self.simulate_sta and self.sta_amp > 0:
            t = torch.linspace(0, seq_len / self.fps, seq_len, device=device).view(1, -1, 1, 1)
            sta_vibration = self.sta_amp * torch.sin(2.0 * 3.14159 * self.sta_freq * t)
            accel = accel + sta_vibration

        return {
            "accel": accel,
            "gyro": gyro,
            "rotations": clean_imu.get("rotations", None),
        }
