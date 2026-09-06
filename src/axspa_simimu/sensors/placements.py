"""
Sensor placement configurations mapping joints/vertices to sensor channels.
"""

# DIP 6-IMU layout (Head, Pelvis, Left Wrist, Right Wrist, Left Lower Leg, Right Lower Leg)
SENSOR_PLACEMENTS_DIP = {
    "num_sensors": 6,
    "names": ["head", "pelvis", "left_wrist", "right_wrist", "left_shank", "right_shank"],
    "joint_indices": [15, 0, 20, 21, 7, 8], # SMPL 24-joint index mapping
}

# Clinical 3-IMU spinal layout for axSpA axial mobility monitoring (C7 Cervical, T12 Thoracolumbar, Sacrum)
SENSOR_PLACEMENTS_SPINE = {
    "num_sensors": 3,
    "names": ["cervical_c7", "thoracolumbar_t12", "sacrum_pelvis"],
    "joint_indices": [9, 6, 0], # Spine3, Spine2, Pelvis
}
