import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from axspa_simimu.models.generator import ConditionalSMPLGenerator
from axspa_simimu.models.discriminator import KinematicDiscriminator
from axspa_simimu.utils.logging import init_logger

def compute_gradient_penalty(D, real_samples, fake_samples, conditions, device):
    """Calculates the gradient penalty (GP) loss for WGAN-GP."""

    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = D(interpolates, conditions)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("Initialising GAN Training...")
    
    # 1. Initialise Logger (WandB setup)
    logger = init_logger(cfg)
    
    # 2. Hardware configuration
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.gpu.get("benchmark_cudnn", True)
    print(f"Using device: {device}")
    
    # 3. Initialise Models
    generator = ConditionalSMPLGenerator(cfg.generator).to(device)
    discriminator = KinematicDiscriminator(cfg.discriminator).to(device)
    
    # 4. Optimisers
    train_cfg = cfg.training
    opt_G = torch.optim.Adam(
        generator.parameters(), 
        lr=train_cfg.learning_rate_g, 
        betas=(train_cfg.adam_b1, train_cfg.adam_b2)
    )
    opt_D = torch.optim.Adam(
        discriminator.parameters(), 
        lr=train_cfg.learning_rate_d, 
        betas=(train_cfg.adam_b1, train_cfg.adam_b2)
    )
    
    # Mixed Precision Scaler for RTX 4050
    amp_enabled = train_cfg.amp.enabled and (device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
        autocast_context = lambda: torch.amp.autocast('cuda', enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        autocast_context = lambda: torch.cuda.amp.autocast(enabled=amp_enabled)
    
    from axspa_simimu.data.amass_dataset import AMASSDataset
    
    # 5. Dataloader Setup
    print("Loading CMU AMASS datasets...")
    
    # Initialize our custom dataset
    dataset = AMASSDataset(cfg)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.gpu.num_workers,
        pin_memory=cfg.gpu.pin_memory
    )
    
    # Hyperparameters from config
    n_critic = train_cfg.n_critic
    lambda_gp = train_cfg.loss_weights.gradient_penalty
    lambda_rom = train_cfg.loss_weights.kinematic_rom
    
    # Checkpointing configuration setup
    ckpt_cfg = train_cfg.get("checkpointing", {})
    save_every_n = ckpt_cfg.get("save_every_n_epochs", 10)
    keep_top_k = ckpt_cfg.get("keep_top_k", 3)
    
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    saved_checkpoints = []  # Tracks (g_loss, filepath) for top-k pruning
    best_g_loss = float("inf")
    
    print(f"Starting WGAN-GP training loop for {train_cfg.epochs} epochs...")
    
    # 6. Training Loop
    for epoch in range(train_cfg.epochs):
        for i, (real_imgs, real_conds) in enumerate(dataloader):
            real_imgs = real_imgs.to(device, non_blocking=True)
            real_conds = real_conds.to(device, non_blocking=True)
            batch_size = real_imgs.size(0)
            
            # Sample random noise and random clinical severity conditions [0, 1]^3 for Fake data
            z = torch.randn(batch_size, cfg.generator.latent_dim, device=device)
            fake_conds = torch.rand(batch_size, cfg.generator.condition_dim, device=device)
            
            # ==========================================
            #  Train Discriminator (Critic)
            # ==========================================
            opt_D.zero_grad()
            
            with autocast_context():
                # Evaluate real data
                real_validity = discriminator(real_imgs, real_conds)
                
                # Generate and evaluate fake data
                fake_imgs = generator(z, fake_conds)
                fake_validity = discriminator(fake_imgs.detach(), fake_conds)
                
                d_adv_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                
            # Compute WGAN-GP gradient penalty in float32 for autograd graph stability
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs.detach(), fake_imgs.detach(), fake_conds.detach(), device
            )
            
            # Total Discriminator Loss
            d_loss = d_adv_loss + lambda_gp * gradient_penalty
                
            scaler.scale(d_loss).backward()
            scaler.step(opt_D)
            scaler.update()
            
            # ==========================================
            #  Train Generator
            # ==========================================
            # Only update the generator every n_critic steps
            if i % n_critic == 0:
                opt_G.zero_grad()
                
                with autocast_context():
                    # Re-evaluate fake data for generator graph
                    fake_imgs = generator(z, fake_conds)
                    fake_validity = discriminator(fake_imgs, fake_conds)
                    
                    # WGAN Generator Adversarial Loss
                    g_adv_loss = -torch.mean(fake_validity)
                    
                    # TODO: Compute Physical Kinematic ROM loss from src/axspa_simimu/losses/
                    # rom_loss = compute_rom_loss(fake_imgs, fake_conds) 
                    rom_loss = torch.tensor(0.0, device=device)
                    
                    # Total Generator Loss
                    g_loss = g_adv_loss + lambda_rom * rom_loss
                    
                scaler.scale(g_loss).backward()
                scaler.step(opt_G)
                scaler.update()
                
            # Log metrics to WandB periodically
            if i % 10 == 0:
                metrics = {
                    "Discriminator/Loss": d_loss.item(),
                    "Discriminator/Gradient_Penalty": gradient_penalty.item(),
                    "Generator/Loss_Adv": g_loss.item() if 'g_loss' in locals() else 0.0,
                }
                logger.log_metrics(metrics, step=epoch * len(dataloader) + i)
                
        # Epoch summary
        print(f"[Epoch {epoch+1}/{train_cfg.epochs}] "
              f"[D loss: {d_loss.item():.4f}] "
              f"[G loss: {g_loss.item() if 'g_loss' in locals() else 0.0:.4f}]")
        
        # Checkpointing logic
        current_g_loss = g_loss.item() if 'g_loss' in locals() else float("inf")
        is_epoch_save = (epoch + 1) % save_every_n == 0 or (epoch + 1) == train_cfg.epochs
        is_best = current_g_loss < best_g_loss
        
        checkpoint_state = {
            "epoch": epoch + 1,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "opt_G_state_dict": opt_G.state_dict(),
            "opt_D_state_dict": opt_D.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "d_loss": d_loss.item(),
            "g_loss": current_g_loss,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        
        # 1. Always save latest checkpoint
        latest_path = os.path.join(ckpt_dir, "checkpoint_latest.pt")
        torch.save(checkpoint_state, latest_path)
        
        # 2. Save best checkpoint based on generator loss
        if is_best:
            best_g_loss = current_g_loss
            best_path = os.path.join(ckpt_dir, "checkpoint_best.pt")
            torch.save(checkpoint_state, best_path)
            print(f"--> Saved new best checkpoint with G loss: {best_g_loss:.4f}")
            
        # 3. Periodic saving & top-k pruning
        if is_epoch_save:
            epoch_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1:04d}.pt")
            torch.save(checkpoint_state, epoch_path)
            print(f"--> Saved periodic checkpoint: {epoch_path}")
            
            saved_checkpoints.append((current_g_loss, epoch_path))
            saved_checkpoints.sort(key=lambda x: x[0])  # Sort by loss ascending
            
            # Prune worse checkpoints exceeding keep_top_k
            while len(saved_checkpoints) > keep_top_k:
                _, path_to_remove = saved_checkpoints.pop(-1)
                if os.path.exists(path_to_remove) and path_to_remove != epoch_path:
                    try:
                        os.remove(path_to_remove)
                    except OSError:
                        pass
                    
            # 4. Log artifact to WandB if enabled
            if getattr(logger, "enabled", False):
                logger.log_artifact(
                    file_or_dir_path=epoch_path,
                    artifact_name=f"{cfg.exp_name}-checkpoint",
                    artifact_type="model",
                    metadata={"epoch": epoch + 1, "g_loss": current_g_loss, "d_loss": d_loss.item()},
                )
        
    logger.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()

