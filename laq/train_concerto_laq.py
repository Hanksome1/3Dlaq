"""
Training script for Concerto-LAQ with Something-Something v2 dataset.

This script:
1. Loads webm videos from the dataset directory
2. Uses VGGT for 3D point cloud generation
3. Uses Concerto for 2D+3D feature extraction
4. Trains the LAQ action quantizer

Usage:
    # Quick test (small model, few videos)
    python train_concerto_laq.py \
        --data_dir /mnt/nfs/eson/dataset/20bn-something-something-v2 \
        --max_videos 100 \
        --batch_size 2 \
        --num_steps 1000

    # Full training
    python train_concerto_laq.py \
        --data_dir /mnt/nfs/eson/dataset/20bn-something-something-v2 \
        --batch_size 8 \
        --num_steps 100000 \
        --use_wandb
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))

from laq_model.concerto_laq import ConcertoLAQ
from laq_model.webm_dataset import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Concerto-LAQ")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .webm video files")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Limit number of videos (for testing)")
    parser.add_argument("--frame_size", type=int, default=224,
                        help="Frame size (must be divisible by 14)")
    parser.add_argument("--frame_offset", type=int, default=5,
                        help="Frame offset between pairs")
    parser.add_argument("--samples_per_video", type=int, default=1,
                        help="Number of samples per video")
    
    # Model
    parser.add_argument("--concerto_model", type=str, default="concerto_base",
                        choices=["concerto_small", "concerto_base", "concerto_large"])
    parser.add_argument("--dim", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--codebook_size", type=int, default=256,
                        help="Number of action codes")
    parser.add_argument("--code_seq_len", type=int, default=4,
                        help="Number of action tokens per frame pair")
    parser.add_argument("--spatial_depth", type=int, default=4,
                        help="Depth of spatial transformer")
    parser.add_argument("--temporal_depth", type=int, default=4,
                        help="Depth of temporal transformer")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--use_tensorboard", action="store_true",
                        help="Use TensorBoard for logging")
    parser.add_argument("--wandb_project", type=str, default="concerto-laq")
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="./runs",
                        help="TensorBoard log directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )
    
    # Initialize TensorBoard
    tb_writer = None
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        
        # Create log directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.wandb_run or f"run_{timestamp}"
        log_path = os.path.join(args.log_dir, run_name)
        os.makedirs(log_path, exist_ok=True)
        
        tb_writer = SummaryWriter(log_path)
        print(f"TensorBoard logs: {log_path}")
        print(f"Run: tensorboard --logdir {args.log_dir}")
        
        # Log hyperparameters
        tb_writer.add_text("hyperparameters", str(vars(args)))
    
    # Create dataloader
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        frame_size=(args.frame_size, args.frame_size),
        frame_offset=args.frame_offset,
        num_samples_per_video=args.samples_per_video,
        max_videos=args.max_videos,
        num_workers=args.num_workers,
    )
    
    # Create model
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)
    
    # Concerto output dim (PTv3 features)
    # Based on Concerto model configs
    concerto_dims = {
        "concerto_small": 512,  # PTv3-small outputs 512
        "concerto_base": 512,   # PTv3-base outputs 512
        "concerto_large": 512,  # PTv3-large outputs 512
    }
    
    model = ConcertoLAQ(
        concerto_model_name=args.concerto_model,
        concerto_dim=concerto_dims[args.concerto_model],
        dim=args.dim,
        quant_dim=32,
        codebook_size=args.codebook_size,
        code_seq_len=args.code_seq_len,
        image_size=args.frame_size,
        feature_size=(args.frame_size // 16, args.frame_size // 16),  # Downsampled
        spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth,
        predictor_depth=4,
        freeze_concerto=True,
        use_precomputed_features=False,
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps)
    
    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["step"]
        print(f"Resumed from step {start_step}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    model.train()
    data_iter = iter(dataloader)
    
    running_loss = 0.0
    running_unique = 0.0
    
    for step in range(start_step, args.num_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        video = batch["video"].to(device)  # [B, C, 2, H, W]
        
        # Forward pass
        try:
            loss, num_unique = model(video=video, step=step)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Warning: OOM at step {step}, skipping batch")
                torch.cuda.empty_cache()
                continue
            raise e
        
        # Backward pass
        loss = loss / args.grad_accum
        loss.backward()
        
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        # Accumulate stats
        running_loss += loss.item() * args.grad_accum
        running_unique += num_unique
        
        # Logging
        if (step + 1) % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            avg_unique = running_unique / args.log_every
            lr = scheduler.get_last_lr()[0]
            
            print(f"Step {step+1}/{args.num_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Unique codes: {avg_unique:.1f} | "
                  f"LR: {lr:.2e}")
            
            if args.use_wandb:
                wandb.log({
                    "loss": avg_loss,
                    "unique_codes": avg_unique,
                    "lr": lr,
                }, step=step+1)
            
            # TensorBoard logging
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", avg_loss, step+1)
                tb_writer.add_scalar("train/unique_codes", avg_unique, step+1)
                tb_writer.add_scalar("train/learning_rate", lr, step+1)
            
            running_loss = 0.0
            running_unique = 0.0
        
        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f"checkpoint_{step+1}.pt"
            )
            torch.save({
                "step": step + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final save
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "step": args.num_steps,
        "model": model.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    if args.use_wandb:
        wandb.finish()
    
    # Close TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
