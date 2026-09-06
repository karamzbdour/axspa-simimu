# axSpA-SimIMU: Biomechanically Constrained Pose Generation and Virtual Wearable Synthesis

This repository provides an end-to-end framework adapting generative 3D pose architectures for clinical wearable research in axial spondyloarthritis (axSpA).

---

## 1. Reference
This project builds upon and extends the baseline 3D human pose augmentation framework introduced in:

> **Wang, X., Mi, Y., & Zhang, X. (2024).** 3D human pose data augmentation using Generative Adversarial Networks for robotic-assisted movement quality assessment. *Frontiers in Neurorobotics*, 18, 1371385.  
> DOI: [10.3389/fnbot.2024.1371385](https://doi.org/10.3389/fnbot.2024.1371385)

---

## 2. Research & Implementation Gap
The base paper presents a GAN architecture focused on unconstrained 3D human pose data augmentation using small sets of robotic-assisted motion capture data, DenseNet feature extraction, and SVM classification evaluated on benchmark datasets (**Human3.6M**, **NTU RGB+D**, **MPI-INF-3DHP**, **HumanEva**). However, several domain-specific limitations prevent its direct use in clinical wearable research:

* **Unconstrained vs. Pathological Kinematics:** The base GAN generates generic human motion from unstructured noise without pathological constraints. It lacks mechanisms to model joint-specific range-of-motion (ROM) restrictions.
* **No Parametric Biomechanical Grounding:** The original framework operates on generic coordinate representations rather than parametric anatomical mesh models (e.g., SMPL/SMPL-X), preventing direct clinical angle clamping.
* **Lack of Sensor Simulation:** The original pipeline terminates at 3D joint/image coordinates. It cannot synthesize wearable sensor streams (IMU signals) from the generated motions.
* **Absence of Downstream Sparse-to-Dense Pipelines:** The base work evaluates pose classification via SVM rather than sparse wearable-to-dense 3D motion reconstruction.

### Addressed in this Repository
* **Conditioned Pathological Synthesis:** Injection of clinical severity vectors $\alpha \in [0, 1]^3$ and kinematic loss constraints to simulate cervical, thoracic, and lumbar spinal stiffness characteristic of axSpA.
* **Virtual Sensor Synthesis Layer:** Kinematic derivation of local-frame tri-axial linear acceleration $\mathbf{a}(t)$ and angular velocity $\boldsymbol{\omega}(t)$ from SMPL vertex trajectories, augmented with realistic sensor drift, noise, and soft-tissue artifact (STA) modeling.
* **Paired Benchmark Dataset Generation:** Construction of synthetic sparse IMU $\rightarrow$ dense 3D pose datasets $(\mathbf{X}_{\text{IMU}}, \mathbf{Y}_{\text{Pose}})$.
* **Hugging Face Hub Streaming & Loading:** Direct integration with Hugging Face datasets for Human3.6M, NTU RGB+D, MPI-INF-3DHP, and HumanEva without requiring manual local directory wrangling.
* **Hardware-Accelerated Configuration:** Out-of-the-box configuration prioritized for NVIDIA GeForce RTX 4050 (AMP fp16 mixed precision, memory pinning, cuDNN benchmarking).
* **Weights & Biases Tracking:** Full experiment tracking, real-time loss and clinical metric curves, 3D visual figures, and model checkpoint artifact logging via WandB.

---

## 3. Modular Configuration System

The repository uses hierarchical YAML configuration (compatible with Hydra / OmegaConf) under `configs/`:

```text
configs/
├── config.yaml                      # Root composition config (RTX 4050 GPU & defaults)
├── logging/                         # Experiment tracking configurations
│   ├── wandb.yaml                   # Weights & Biases online tracking & artifact versioning
│   └── disabled.yaml                # Offline / local logging only
├── dataset/                         # Hugging Face dataset configs (Paper benchmarks)
│   ├── human3.6m.yaml               # Human3.6M via Hugging Face Hub
│   ├── ntu_rgbd.yaml                # NTU RGB+D via Hugging Face Hub
│   ├── mpi_inf_3dhp.yaml            # MPI-INF-3DHP via Hugging Face Hub
│   └── humaneva.yaml                # HumanEva via Hugging Face Hub
├── pathology/                       # Clinical stiffness & ROM limits
│   ├── healthy_control.yaml         # Unconstrained natural ROM
│   ├── axspa_mild.yaml              # Partial lumbar/thoracic stiffness (alpha = [0.35, 0.40, 0.50])
│   └── axspa_severe.yaml            # Severe ankylosis & spinal fusion (alpha = [0.85, 0.90, 0.95])
├── sensor/                          # Virtual wearable layout & noise profiles
│   ├── dip_6imu.yaml                # DIP 6-IMU layout (head, pelvis, wrists, shanks)
│   ├── clinical_spine_3imu.yaml     # 3-IMU spinal layout (C7, T12, Sacrum)
│   └── noise_profile/
│       ├── clean.yaml               # Ideal physics simulation (zero noise)
│       └── realistic_wearable.yaml  # White noise + random-walk bias drift + soft-tissue artifacts
├── model/                           # Architecture specifications
│   ├── generator_smpl.yaml          # Conditional SMPL GAN generator
│   ├── discriminator.yaml           # Kinematic discriminator
│   └── downstream/
│       ├── transformer.yaml         # Spatial-temporal pose Transformer
│       ├── tcn.yaml                 # Temporal Convolutional Network
│       └── gnn.yaml                 # Graph Neural Network
└── training/                        # Training hyperparameters
    ├── gan_wgan_gp.yaml             # W-GAN-GP optimization & loss weighting
    └── downstream_recon.yaml        # Sparse-to-dense reconstruction training
```

### Overriding Configs from Command Line
You can easily switch datasets, models, pathology profiles, sensor setups, or logging modes:

```bash
# 1. Train GAN with Weights & Biases logging enabled (default)
python scripts/train_gan.py dataset=ntu_rgbd pathology=axspa_severe

# 2. Run an offline experiment without internet connection
python scripts/train_gan.py logging.mode=offline

# 3. Disable Weights & Biases logging completely
python scripts/train_gan.py logging=disabled

# 4. Synthesize 3-IMU clinical spine dataset with realistic wearable noise
python scripts/synthesize_imu_dataset.py sensor=clinical_spine_3imu sensor/noise_profile=realistic_wearable

# 5. Run downstream reconstruction using the Transformer architecture
python scripts/train_downstream.py model/downstream=transformer
```

---

## 4. Weights & Biases (WandB) Setup

### Authentication
Log in to your Weights & Biases account:
```bash
wandb login
```
Alternatively, set the environment variable:
```bash
export WANDB_API_KEY="your_api_key_here"  # On Linux/macOS
$env:WANDB_API_KEY="your_api_key_here"    # On Windows PowerShell
```

### Project & Entity Configuration
In [`configs/logging/wandb.yaml`](file:///C:/Users/HP/OneDrive/Documents/GitHub/pose-estimation-enrichment/configs/logging/wandb.yaml), you can configure:
* `project`: WandB project name (defaults to `"axspa-simimu"`).
* `entity`: Your personal username or team name.
* `tags`: Run tags (e.g. `["rtx4050", "human3.6m", "transformer"]`).
* `log_model_artifacts`: Set `true` to version trained `.pt` checkpoints directly to the WandB registry.

---

## 5. Repository Structure

```text
pose-estimation-enrichment/
├── configs/                         # Modular configuration hierarchy
├── src/
│   └── axspa_simimu/                # Core Python package
│       ├── biomechanics/            # Clinical ROM constraints, BASMI index & SMPL kinematics
│       │   ├── rom_constraints.py   # Pathological ROM limits & angular scaling
│       │   ├── clinical_indices.py  # BASMI scores & spinal stiffness metrics
│       │   └── smpl_wrapper.py      # Forward kinematics & 6D rotation conversions
│       ├── sensors/                 # Physics kinematics & sensor simulation
│       │   ├── virtual_imu.py       # Forward kinematic accel a(t) & gyro omega(t) synthesis
│       │   ├── noise_models.py      # Gaussian noise, drift & soft-tissue artifacts
│       │   └── placements.py        # Anatomical sensor placement mappings
│       ├── models/                  # Neural network model definitions
│       ├── losses/                  # Kinematic, smoothness & adversarial losses
│       ├── data/                    # Hugging Face loaders & synthetic dataset handlers
│       ├── metrics/                 # MPJPE, sensor RMSE & clinical score metrics
│       └── utils/                   # WandbLogger, experiment logging & visualizers
├── scripts/                         # CLI execution entrypoints
├── tests/                           # Unit & integration tests
├── pyproject.toml                   # Build & packaging configuration
├── environment.yml                  # Conda environment definition
└── README.md
```

---

## 6. Installation & Setup

### Environment Setup (Conda / Mamba)
```bash
conda env create -f environment.yml
conda activate axspa-simimu
pip install -e .
```

### Hugging Face Datasets Authentication (Optional for gated benchmarks)
```bash
export HF_TOKEN="your_huggingface_token"
```