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
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("Initialising GAN Training...")
    
    # 1. Initialise Logger (WandB setup)
    logger = init_logger(cfg)
    
    # 2. Hardware configuration
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Initialise Models
    generator = ConditionalSMPLGenerator(cfg.model).to(device)
    discriminator = KinematicDiscriminator(cfg.model).to(device)
    
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
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.amp.enabled)
    
    # 5. Dataloader Setup
    # TODO: Replace this with your actual HuggingFace dataset logic 
    # (e.g. from src.axspa_simimu.data.h36m_loader)
    print("Loading datasets...")
    B, seq_length = train_cfg.batch_size, cfg.model.seq_length
    feature_dim = 144 if cfg.model.representation == "rotation_6d" else 72
    
    # Dummy data representing healthy benchmark data (c = [0,0,0])
    dummy_real_data = torch.randn(20 * B, seq_length, feature_dim) 
    dummy_real_cond = torch.zeros(20 * B, cfg.model.condition_dim) 
    
    dataloader = DataLoader(
        TensorDataset(dummy_real_data, dummy_real_cond), 
        batch_size=B, 
        shuffle=True, 
        num_workers=cfg.gpu.num_workers,
        pin_memory=cfg.gpu.pin_memory
    )
    
    # Hyperparameters from config
    n_critic = train_cfg.n_critic
    lambda_gp = train_cfg.loss_weights.gradient_penalty
    lambda_rom = train_cfg.loss_weights.kinematic_rom
    
    print(f"Starting WGAN-GP training loop for {train_cfg.epochs} epochs...")
    
    # 6. Training Loop
    for epoch in range(train_cfg.epochs):
        for i, (real_imgs, real_conds) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            real_conds = real_conds.to(device)
            batch_size = real_imgs.size(0)
            
            # Sample random noise and random clinical severity conditions [0, 1]^3 for Fake data
            z = torch.randn(batch_size, cfg.model.latent_dim, device=device)
            fake_conds = torch.rand(batch_size, cfg.model.condition_dim, device=device)
            
            # ==========================================
            #  Train Discriminator (Critic)
            # ==========================================
            opt_D.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=train_cfg.amp.enabled):
                # Evaluate real data
                real_validity = discriminator(real_imgs, real_conds)
                
                # Generate and evaluate fake data
                fake_imgs = generator(z, fake_conds)
                fake_validity = discriminator(fake_imgs.detach(), fake_conds)
                
                # Compute WGAN-GP gradient penalty
                gradient_penalty = compute_gradient_penalty(
                    discriminator, real_imgs.data, fake_imgs.data, fake_conds.data, device
                )
                
                # WGAN Discriminator Loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                
            scaler.scale(d_loss).backward()
            scaler.step(opt_D)
            scaler.update()
            
            # ==========================================
            #  Train Generator
            # ==========================================
            # Only update the generator every n_critic steps
            if i % n_critic == 0:
                opt_G.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=train_cfg.amp.enabled):
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
        
        # TODO: Add model checkpointing logic based on cfg.training.checkpointing
        
    logger.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()

