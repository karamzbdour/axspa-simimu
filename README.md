# axSpA-SimIMU: Biomechanically Constrained Pose Generation and Virtual Wearable Synthesis

A framework for generative 3D human pose synthesis and virtual inertial sensor (IMU) simulation designed for clinical wearable research in axial spondyloarthritis (axSpA).

---

## Generated Motion Visualisations

| Healthy Control | Pathological (axSpA) |
| :---: | :---: |
| ![Healthy Control Motion](outputs/visualizations/Healthy_Control_sample_01.gif) | ![Severe axSpA Motion](outputs/visualizations/Severe_axSpA_sample_01.gif)<br>|
| **Healthy Control (Unconstrained Kinematics)**<br>Natural spinal flexibility and full range of motion. | **Severe axSpA (Spinal Stiffness & Fusion)**<br>Restricted lumbar, thoracic, and cervical mobility. |

---

## GAN Architecture

The generative model uses a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** conditioned on clinical severity to synthesise temporal sequences of SMPL body poses:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                      Generator                         │
Latent Vector z ─► [Concat] ─► 2-Layer GRU ─► Multi-Layer Perceptron (MLP)   ──► 6D Pose Sequence
Condition α    ─► │ (128+3)       (Hidden: 256)   (256→512→512→256→144)    |    (T=60, 24 joints × 6)
                  └────────────────────────────────────────────────────────┘

                  ┌────────────────────────────────────────────────────────┐
                  │                Kinematic Discriminator                 │
Pose Sequence   ─► [Concat] ─► 1D Conv Layers (64→128→256→512) ─► Linear     ──► Scalar Score
Condition α    ─► │ (144+3)       (Kernel: 5, Stride: 2, Spectral Norm)    |
                  └────────────────────────────────────────────────────────┘
```

* **Generator (`ConditionalSMPLGenerator`)**:
  * **Input**: Concatenated latent noise $z \sim \mathcal{N}(0, I)^{128}$ and clinical condition vector $\alpha \in [0, 1]^3$.
  * **Temporal Backbone**: 2-layer Gated Recurrent Unit (GRU) with hidden dimension 256 across $T = 60$ frames (1.0 second @ 60 Hz).
  * **Projection MLP**: Linear layers `[256 -> 512 -> 512 -> 256 -> 144]` with Layer Normalisation, LeakyReLU ($\alpha = 0.2$), and Dropout ($0.1$).
  * **Output**: Continuous 6D rotation representation for 24 standard SMPL joints ($24 \times 6 = 144$ dimensions per frame).

* **Discriminator / Critic (`KinematicDiscriminator`)**:
  * **Input**: Pose sequence concatenated with temporal condition vector ($147$ channels $\times 60$ frames).
  * **Architecture**: 4-layer 1D Temporal Convolutional Network (channel progression: `64 -> 128 -> 256 -> 512`, kernel size 5, stride 2) with Spectral Normalisation, LeakyReLU ($\alpha = 0.2$), and Dropout ($0.2$).
  * **Output**: Scalar Wasserstein validity score via a final linear projection.

* **Training & Optimisation**:
  * **Loss**: WGAN-GP adversarial objective with Gradient Penalty weight $\lambda_{\text{GP}} = 10.0$.
  * **Optimisers**: Adam ($lr_G = 1\times 10^{-4}$, $lr_D = 4\times 10^{-4}$, $\beta_1 = 0.5, \beta_2 = 0.999$).
  * **Critic Updates**: $n_{\text{critic}} = 5$ discriminator steps per generator step.
  * **Precision**: Automatic Mixed Precision (AMP FP16) for GPU acceleration.

---

## Dataset

The framework trains on the **AMASS (Archive of Motion Capture as Surface Shapes)** dataset, using the **CMU MoCap** subset:

* **Format**: SMPL/SMPL-H parameter `.npz` files.
* **Joint Representation**: 24 standard body joints extracted from the first 72 axis-angle parameters and converted to continuous 6D rotation matrices ($24 \times 6 = 144$ dimensions).
* **Sampling Rate**: Resampled/subsampled to a uniform **60 Hz**.
* **Windowing**: Sliced into overlapping sliding windows of **60 frames** (1.0 second duration) with a stride of **30 frames**.

---

## Modular Configuration System

The repository uses **Hydra / OmegaConf** for hierarchical configuration under `configs/`:

* **`configs/config.yaml`**: Primary entry point composing default sub-configs and hardware acceleration settings (CUDA, AMP fp16).
* **`configs/model/`**: Generator (`generator_smpl.yaml`) and Discriminator (`discriminator.yaml`) architectures.
* **`configs/dataset/`**: Dataset configurations (`cmu_amass.yaml`, benchmark formats).
* **`configs/training/`**: Training hyperparameters and loss weightings (`gan_wgan_gp.yaml`).
* **`configs/pathology/`**: Clinical stiffness and ROM presets (`healthy_control.yaml`, `axspa_mild.yaml`, `axspa_severe.yaml`).
* **`configs/sensor/`**: Virtual IMU layouts (e.g. 6-IMU DIP, 3-IMU Spine) and noise profiles (`realistic_wearable.yaml`, `clean.yaml`).
* **`configs/logging/`**: Weights & Biases (`wandb.yaml`) and local/disabled tracking (`disabled.yaml`).

### CLI Overrides
Any configuration parameter can be overridden dynamically from the command line:

```bash
# Train with custom epochs and batch size
python scripts/train_gan.py training.epochs=300 training.batch_size=64

# Train with disabled WandB logging (local console only)
python scripts/train_gan.py logging=disabled

# Specify custom dataset path
python scripts/train_gan.py dataset.data_dir="data/amass/CMU"
```

---

## Project Directory

```text
pose-estimation-enrichment/
├── configs/                         # Modular Hydra YAML configs
│   ├── config.yaml                  # Root composition config
│   ├── dataset/                     # Dataset settings (cmu_amass.yaml)
│   ├── logging/                     # WandB & console logger configs
│   ├── model/                       # Generator & discriminator architectures
│   ├── pathology/                   # Clinical ROM & stiffness definitions
│   ├── sensor/                      # Virtual IMU layouts & noise profiles
│   └── training/                    # WGAN-GP training hyperparameters
├── src/
│   └── axspa_simimu/                # Core package
│       ├── biomechanics/            # Forward kinematics, ROM bounds, BASMI metrics
│       │   ├── clinical_indices.py  # Clinical mobility scores & stiffness indices
│       │   ├── rom_constraints.py   # Range of Motion limits & scaling functions
│       │   └── smpl_wrapper.py      # SMPL 24-joint forward kinematics & 6D rotations
│       ├── data/                    # Dataset loaders & windowing (AMASSDataset)
│       ├── models/                  # Generator & Discriminator PyTorch modules
│       ├── sensors/                 # Virtual IMU synthesis, noise, and placement models
│       └── utils/                   # Logging, WandB integration, and utilities
├── scripts/
│   ├── train_gan.py                 # WGAN-GP training entrypoint
│   └── visualise_poses.py           # 3D skeleton rendering, GIF export & keyframe plots
├── checkpoints/                     # Saved model checkpoints (.pt)
├── outputs/                         # Run outputs, logs, and visualisation GIFs
├── data/                            # Local dataset directory (e.g. data/amass/CMU)
├── environment.yml                  # Conda environment specification
├── pyproject.toml                   # Package installation metadata
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
# Clone repository and create conda environment
conda env create -f environment.yml
conda activate axspa-simimu

# Install axspa_simimu package in editable mode
pip install -e .
```

### 2. Training the GAN

```bash
# Train WGAN-GP on CMU AMASS dataset
python scripts/train_gan.py
```

### 3. Visualising & Generating Poses

```bash
# Generate 3D motion GIFs and keyframe progression plots from checkpoint
python scripts/visualise_poses.py --checkpoint checkpoints/checkpoint_best.pt --compare_pathology
```