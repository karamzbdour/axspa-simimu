"""
Virtual IMU synthesis, physical kinematics, and sensor noise simulation.
"""

from axspa_simimu.sensors.virtual_imu import VirtualIMUSimulator
from axspa_simimu.sensors.noise_models import IMUNoiseSimulator
from axspa_simimu.sensors.placements import SENSOR_PLACEMENTS_DIP, SENSOR_PLACEMENTS_SPINE

__all__ = [
    "VirtualIMUSimulator",
    "IMUNoiseSimulator",
    "SENSOR_PLACEMENTS_DIP",
    "SENSOR_PLACEMENTS_SPINE",
]
