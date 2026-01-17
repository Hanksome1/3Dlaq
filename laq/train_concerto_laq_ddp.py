"""
Multi-GPU Training script for Concerto-LAQ with DistributedDataParallel.

This script supports training on multiple GPUs using PyTorch DDP.

Usage:
    # Single GPU
    python train_concerto_laq_ddp.py --data_dir /path/to/data --batch_size 2

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 train_concerto_laq_ddp.py \
        --data_dir /path/to/data --batch_size 4
    
    # With specific GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_concerto_laq_ddp.py \
        --data_dir /path/to/data --batch_size 4
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))

from laq_model.concerto_laq import ConcertoLAQ
from laq_model.webm_dataset import WebmVideoDataset


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Train Concerto-LAQ (Multi-GPU)")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .webm video files")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Limit number of videos (for testing)")
    parser.add_argument("--frame_size", type=int, default=256,
                        help="Frame size (must be divisible by 14)")
    parser.add_argument("--frame_offset", type=int, default=30,
                        help="Frame offset between pairs (LAQ uses 30)")
    parser.add_argument("--samples_per_video", type=int, default=1,
                        help="Number of samples per video")
    
    # Model - LAQ paper defaults
    parser.add_argument("--concerto_model", type=str, default="concerto_base",
                        choices=["concerto_small", "concerto_base", "concerto_large"])
    parser.add_argument("--dim", type=int, default=1024,
                        help="Model dimension (LAQ uses 1024)")
    parser.add_argument("--quant_dim", type=int, default=32,
                        help="Quantization embedding dimension (LAQ uses 32)")
    parser.add_argument("--codebook_size", type=int, default=8,
                        help="Number of action codes (LAQ uses 8)")
    parser.add_argument("--code_seq_len", type=int, default=4,
                        help="Number of action tokens per frame pair (LAQ uses 4)")
    parser.add_argument("--spatial_depth", type=int, default=8,
                        help="Depth of spatial transformer (LAQ uses 8)")
    parser.add_argument("--temporal_depth", type=int, default=8,
                        help="Depth of temporal transformer (LAQ uses 8)")
    parser.add_argument("--heads", type=int, default=16,
                        help="Number of attention heads (LAQ uses 16)")
    parser.add_argument("--dim_head", type=int, default=64,
                        help="Dimension per head (LAQ uses 64)")
    
    # Training - LAQ paper defaults
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU (total = batch_size * world_size)")
    parser.add_argument("--num_steps", type=int, default=100005,
                        help="Training steps (LAQ uses ~100K)")
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
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="concerto-laq")
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="./runs")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    if is_main_process():
        print(f"World size: {world_size} GPUs")
        print(f"Using device: {device}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logging (main process only)
    tb_writer = None
    if is_main_process():
        if args.use_wandb:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run,
                config=vars(args),
            )
        
        if args.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = args.wandb_run or f"run_{timestamp}"
            log_path = os.path.join(args.log_dir, run_name)
            os.makedirs(log_path, exist_ok=True)
            tb_writer = SummaryWriter(log_path)
            print(f"TensorBoard logs: {log_path}")
            print(f"Run: tensorboard --logdir {args.log_dir}")
    
    # Create dataset
    if is_main_process():
        print("\n" + "="*60)
        print("Loading Dataset")
        print("="*60)
    
    dataset = WebmVideoDataset(
        data_dir=args.data_dir,
        frame_size=(args.frame_size, args.frame_size),
        frame_offset=args.frame_offset,
        num_samples_per_video=args.samples_per_video,
        max_videos=args.max_videos,
    )
    
    # Distributed sampler
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    if is_main_process():
        print(f"Total samples: {len(dataset)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
    
    # Create model
    if is_main_process():
        print("\n" + "="*60)
        print("Creating Model")
        print("="*60)
    
    # Concerto output dim
    concerto_dims = {
        "concerto_small": 512,
        "concerto_base": 512,
        "concerto_large": 512,
    }
    
    model = ConcertoLAQ(
        concerto_model_name=args.concerto_model,
        concerto_dim=concerto_dims[args.concerto_model],
        dim=args.dim,
        quant_dim=args.quant_dim,
        codebook_size=args.codebook_size,
        code_seq_len=args.code_seq_len,
        image_size=args.frame_size,
        feature_size=(args.frame_size // 16, args.frame_size // 16),
        spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth,
        predictor_depth=4,
        dim_head=args.dim_head,
        heads=args.heads,
        freeze_concerto=True,
        use_precomputed_features=False,
    )
    
    model = model.to(device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    if is_main_process():
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
        if is_main_process():
            print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Handle DDP state dict
        state_dict = checkpoint["model"]
        if world_size > 1:
            # Add 'module.' prefix if not present
            if not any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        else:
            # Remove 'module.' prefix if present
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["step"]
        if is_main_process():
            print(f"Resumed from step {start_step}")
    
    # Training loop
    if is_main_process():
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
    
    model.train()
    running_loss = 0.0
    running_unique = 0.0
    
    for step in range(start_step, args.num_steps):
        # Set epoch for distributed sampler
        if sampler is not None:
            sampler.set_epoch(step)
        
        # Get batch
        data_iter = iter(dataloader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        video = batch["video"].to(device)
        
        # Forward pass
        try:
            if world_size > 1:
                loss, num_unique, metrics = model.module(video=video, step=step)
            else:
                loss, num_unique, metrics = model(video=video, step=step)
        except RuntimeError as e:
            if "out of memory" in str(e):
                if is_main_process():
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
        
        # Accumulate metrics
        if 'running_metrics' not in locals():
            running_metrics = {k: 0.0 for k in metrics.keys() if isinstance(metrics[k], (int, float, torch.Tensor))}
        for k, v in metrics.items():
            if k in running_metrics:
                running_metrics[k] += v.item() if isinstance(v, torch.Tensor) else v
        
        # Logging (main process only)
        if (step + 1) % args.log_every == 0 and is_main_process():
            avg_loss = running_loss / args.log_every
            avg_unique = running_unique / args.log_every
            lr = scheduler.get_last_lr()[0]
            avg_metrics = {k: v / args.log_every for k, v in running_metrics.items()}
            
            print(f"Step {step+1}/{args.num_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Recon: {avg_metrics.get('reconstruction_loss', 0):.4f} | "
                  f"Commit: {avg_metrics.get('commitment_loss', 0):.4f} | "
                  f"Unique: {avg_unique:.1f} | "
                  f"Util: {avg_metrics.get('codebook_utilization', 0):.2%} | "
                  f"Perp: {avg_metrics.get('perplexity', 0):.1f} | "
                  f"LR: {lr:.2e}")
            
            if args.use_wandb:
                import wandb
                wandb.log({
                    "loss": avg_loss,
                    "reconstruction_loss": avg_metrics.get('reconstruction_loss', 0),
                    "commitment_loss": avg_metrics.get('commitment_loss', 0),
                    "unique_codes": avg_unique,
                    "codebook_utilization": avg_metrics.get('codebook_utilization', 0),
                    "perplexity": avg_metrics.get('perplexity', 0),
                    "lr": lr,
                }, step=step+1)
            
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", avg_loss, step+1)
                tb_writer.add_scalar("train/reconstruction_loss", avg_metrics.get('reconstruction_loss', 0), step+1)
                tb_writer.add_scalar("train/commitment_loss", avg_metrics.get('commitment_loss', 0), step+1)
                tb_writer.add_scalar("train/unique_codes", avg_unique, step+1)
                tb_writer.add_scalar("train/codebook_utilization", avg_metrics.get('codebook_utilization', 0), step+1)
                tb_writer.add_scalar("train/perplexity", avg_metrics.get('perplexity', 0), step+1)
                tb_writer.add_scalar("train/learning_rate", lr, step+1)
            
            running_loss = 0.0
            running_unique = 0.0
            running_metrics = {k: 0.0 for k in metrics.keys() if isinstance(metrics[k], (int, float, torch.Tensor))}
        
        # Save checkpoint (main process only)
        if (step + 1) % args.save_every == 0 and is_main_process():
            checkpoint_path = os.path.join(
                args.output_dir,
                f"checkpoint_{step+1}.pt"
            )
            # Save without 'module.' prefix for easier loading
            state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
            torch.save({
                "step": step + 1,
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final save (main process only)
    if is_main_process():
        final_path = os.path.join(args.output_dir, "final_model.pt")
        state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
        torch.save({
            "step": args.num_steps,
            "model": state_dict,
            "args": vars(args),
        }, final_path)
        print(f"\nTraining complete! Final model saved to {final_path}")
        
        if args.use_wandb:
            import wandb
            wandb.finish()
        
        if tb_writer is not None:
            tb_writer.close()
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
