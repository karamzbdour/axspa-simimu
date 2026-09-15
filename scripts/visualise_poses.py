"""
Script for generating and visualizing 3D SMPL pose sequences from a trained GAN checkpoint.
Generates animated 3D skeleton GIFs and multi-frame progression plots.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from omegaconf import OmegaConf

from axspa_simimu.models.generator import ConditionalSMPLGenerator
from axspa_simimu.biomechanics.smpl_wrapper import SMPLSkeletonKinematics, rotation_6d_to_matrix


# SMPL 24-joint kinematic bone connections for skeleton wireframe rendering
SMPL_BONES = [
    (0, 1), (0, 2), (0, 3),      # Pelvis to L_Hip, R_Hip, Spine1
    (1, 4), (4, 7), (7, 10),     # Left leg: L_Hip -> L_Knee -> L_Ankle -> L_Foot
    (2, 5), (5, 8), (8, 11),     # Right leg: R_Hip -> R_Knee -> R_Ankle -> R_Foot
    (3, 6), (6, 9),              # Spine1 -> Spine2 -> Spine3
    (9, 12), (12, 15),           # Spine3 -> Neck -> Head
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), # Left arm & collar
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23), # Right arm & collar
]

# Canonical anatomical rest-pose offsets for standard 24 SMPL joints (in meters)
CANONICAL_OFFSETS = torch.tensor([
    [0.0, 0.0, 0.0],          # 0: Pelvis
    [0.07, -0.09, 0.0],       # 1: L_Hip
    [-0.07, -0.09, 0.0],      # 2: R_Hip
    [0.0, 0.12, 0.0],         # 3: Spine1
    [0.0, -0.38, 0.0],        # 4: L_Knee
    [0.0, -0.38, 0.0],        # 5: R_Knee
    [0.0, 0.12, 0.0],         # 6: Spine2
    [0.0, -0.39, 0.0],        # 7: L_Ankle
    [0.0, -0.39, 0.0],        # 8: R_Ankle
    [0.0, 0.12, 0.0],         # 9: Spine3
    [0.0, -0.08, 0.15],       # 10: L_Foot
    [0.0, -0.08, 0.15],       # 11: R_Foot
    [0.0, 0.15, 0.0],         # 12: Neck
    [0.08, 0.10, -0.02],      # 13: L_Collar
    [-0.08, 0.10, -0.02],     # 14: R_Collar
    [0.0, 0.12, 0.0],         # 15: Head
    [0.12, 0.0, 0.0],         # 16: L_Shoulder
    [-0.12, 0.0, 0.0],        # 17: R_Shoulder
    [0.25, 0.0, 0.0],         # 18: L_Elbow
    [-0.25, 0.0, 0.0],        # 19: R_Elbow
    [0.24, 0.0, 0.0],         # 20: L_Wrist
    [-0.24, 0.0, 0.0],        # 21: R_Wrist
    [0.08, 0.0, 0.0],         # 22: L_Hand
    [-0.08, 0.0, 0.0],        # 23: R_Hand
], dtype=torch.float32)


def render_skeleton_frame(positions_3d, ax, title="3D Pose", elev=15, azim=135, radius=0.9):
def render_skeleton_frame(positions_3d, ax, title="3D Pose", elev=15, azim=45, radius=0.9):
    """Draws a single 3D skeleton frame onto a matplotlib 3D axis."""
    ax.cla()
    pts = positions_3d.numpy() if isinstance(positions_3d, torch.Tensor) else positions_3d
    
    # Root centering at Pelvis
    root = pts[0]
    
    # Draw bones
    for p1, p2 in SMPL_BONES:
        ax.plot(
            [pts[p1, 0], pts[p2, 0]],
            [pts[p1, 2], pts[p2, 2]], # Swap Y and Z for natural upright rendering
            [pts[p1, 1], pts[p2, 1]],
            [pts[p1, 2], pts[p2, 2]],
            color="#2b5c8f",
            linewidth=2.5,
            alpha=0.85
        )
        
    # Draw joint nodes
    ax.scatter(
        pts[:, 0], pts[:, 2], pts[:, 1],
        pts[:, 0], pts[:, 1], pts[:, 2],
        c="#d9534f",
        s=25,
        depthshade=True,
        alpha=0.95
    )
    
    # Equal 3D aspect ratio bounding box
    ax.set_xlim([root[0] - radius, root[0] + radius])
    ax.set_ylim([root[2] - radius, root[2] + radius])
    ax.set_zlim([root[1] - radius, root[1] + radius])
    ax.set_ylim([root[1] - radius, root[1] + radius])
    ax.set_zlim([root[2] - radius, root[2] + radius])
    
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Z (m)", fontsize=8)
    ax.set_zlabel("Y (Up, m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (Up, m)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)


def create_animation_gif(positions_seq, output_path, fps=15, title="Synthesised Motion", elev=15, azim=135):
def create_animation_gif(positions_seq, output_path, fps=15, title="Synthesised Motion", elev=15, azim=45):
    """
    Renders a sequence of 3D joint positions (SeqLen, 24, 3) into an animated GIF.
    """
    seq_len = positions_seq.shape[0]
    frames = []
    
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    
    for t in range(seq_len):
        render_skeleton_frame(
            positions_seq[t],
            ax,
            title=f"{title} (Frame {t+1:02d}/{seq_len})",
            elev=elev,
            azim=azim + (t * 0.5) # Subtle camera rotation
        )
        fig.tight_layout()
        fig.canvas.draw()
        
        # Extract RGB buffer compatible with all modern matplotlib versions
        rgba = np.asarray(fig.canvas.buffer_rgba())
        image = rgba[:, :, :3].copy()
        frames.append(image)
        
    plt.close(fig)
    
    # Save GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"--> Saved 3D animation GIF to: {output_path}")


def create_progression_plot(positions_seq, output_path, num_keyframes=6, title="Pose Progression", elev=15, azim=135):
def create_progression_plot(positions_seq, output_path, num_keyframes=6, title="Pose Progression", elev=15, azim=45):
    """
    Renders keyframe snapshots across time into a single comparison grid.
    """
    seq_len = positions_seq.shape[0]
    indices = np.linspace(0, seq_len - 1, num_keyframes, dtype=int)
    
    fig = plt.figure(figsize=(3.5 * num_keyframes, 4), dpi=120)
    
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(1, num_keyframes, i + 1, projection="3d")
        render_skeleton_frame(
            positions_seq[idx],
            ax,
            title=f"Frame {idx+1}",
            elev=elev,
            azim=azim
        )
        
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved keyframe progression plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate & visualize 3D SMPL motions from GAN checkpoints.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best.pt", help="Path to checkpoint .pt")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations", help="Output directory for GIFs & plots")
    parser.add_argument("--condition", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Clinical condition [cervical, thoracic, lumbar] severity in [0, 1]")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of distinct motion samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--compare_pathology", action="store_true", help="Generate side-by-side comparison between Healthy [0,0,0] and Severe axSpA [1,1,1]")
    parser.add_argument("--fps", type=int, default=15, help="Playback frame rate for the GIF (lower value = slower animation, default: 15)")
    parser.add_argument("--elev", type=float, default=15.0, help="Camera elevation angle in degrees (default: 15.0)")
    parser.add_argument("--azim", type=float, default=135.0, help="Camera azimuth angle in degrees (default: 135.0, rotated 90 deg anticlockwise from 45 deg)")
    parser.add_argument("--azim", type=float, default=45.0, help="Camera azimuth angle in degrees (default: 45.0 isometric view)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}'")
        
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    cfg = OmegaConf.create(checkpoint["config"])
    print(f"Checkpoint Epoch: {checkpoint.get('epoch', 'N/A')} | G Loss: {checkpoint.get('g_loss', 'N/A'):.4f}")
    
    # 2. Build Generator & Load Weights
    generator = ConditionalSMPLGenerator(cfg.generator)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    
    # 3. Setup Forward Kinematics
    fk = SMPLSkeletonKinematics(CANONICAL_OFFSETS)
    
    conditions_to_run = []
    if args.compare_pathology:
        conditions_to_run.append(("Healthy_Control", [0.0, 0.0, 0.0]))
        conditions_to_run.append(("Severe_axSpA", [1.0, 1.0, 1.0]))
    else:
        label = f"Condition_C{args.condition[0]:.1f}_T{args.condition[1]:.1f}_L{args.condition[2]:.1f}"
        conditions_to_run.append((label, args.condition))
        
    for cond_name, cond_vec in conditions_to_run:
        print(f"\n--- Generating for condition: {cond_name} (Severity: {cond_vec}) ---")
        c_tensor = torch.tensor([cond_vec], dtype=torch.float32)
        
        for s in range(args.num_samples):
            z = torch.randn(1, cfg.generator.latent_dim)
            with torch.no_grad():
                # Generate 6D poses: (1, SeqLen, 144)
                fake_6d = generator(z, c_tensor)
                
                # Convert 6D -> (1, SeqLen, 24, 3, 3) Rotation Matrices
                rot_matrices = rotation_6d_to_matrix(fake_6d.view(1, cfg.generator.seq_length, 24, 6))
                
                # Compute 3D Cartesian positions: (1, SeqLen, 24, 3)
                global_positions, _ = fk.forward_kinematics(rot_matrices)
                
            pos_seq = global_positions.squeeze(0) # (SeqLen, 24, 3)
            
            # Export GIF animation
            gif_path = os.path.join(args.output_dir, f"{cond_name}_sample_{s+1:02d}.gif")
            create_animation_gif(pos_seq, gif_path, fps=args.fps, title=f"{cond_name} (Sample {s+1})", elev=args.elev, azim=args.azim)
            
            # Export Static Progression Plot
            png_path = os.path.join(args.output_dir, f"{cond_name}_sample_{s+1:02d}_progression.png")
            create_progression_plot(pos_seq, png_path, title=f"3D Motion Progression - {cond_name} (Sample {s+1})", elev=args.elev, azim=args.azim)
            
            # Export Raw Numpy Arrays for downstream evaluation
            npz_path = os.path.join(args.output_dir, f"{cond_name}_sample_{s+1:02d}_poses.npz")
            np.savez(
                npz_path,
                positions_3d=pos_seq.numpy(),
                rot_matrices=rot_matrices.squeeze(0).numpy(),
                condition=np.array(cond_vec),
                seq_length=cfg.generator.seq_length,
                fps=cfg.dataset.sampling_rate_hz
            )
            print(f"--> Saved raw 3D pose data to: {npz_path}")

    print("\nVisualization generation complete!")

if __name__ == "__main__":
    main()
