"""
Training script for Concerto-LAQ.

Example usage:
    # Train with raw video (slower, on-the-fly feature extraction)
    python train_concerto_laq.py --folder /path/to/videos --batch_size 32

    # Train with pre-computed features (faster)
    python train_concerto_laq.py --feature_cache_dir /path/to/features --batch_size 64

    # With custom hyperparameters
    python train_concerto_laq.py --folder /path/to/videos --codebook_size 16 --code_seq_len 4
"""

import argparse
from pathlib import Path

from laq_model.concerto_laq import ConcertoLAQ
from laq_model.concerto_trainer import ConcertoLAQTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Concerto-LAQ model')
    
    # Data arguments
    parser.add_argument('--folder', type=str, default='',
                        help='Path to video folder (for raw video training)')
    parser.add_argument('--feature_cache_dir', type=str, default=None,
                        help='Path to pre-computed Concerto features')
    parser.add_argument('--offset', type=int, default=30,
                        help='Frame offset between training pairs')
    
    # Model arguments
    parser.add_argument('--concerto_model', type=str, default='concerto_base',
                        choices=['concerto_small', 'concerto_base', 'concerto_large'],
                        help='Concerto model size')
    parser.add_argument('--concerto_dim', type=int, default=256,
                        help='Concerto output dimension (256 for base, 512 for large)')
    parser.add_argument('--dim', type=int, default=1024,
                        help='Internal model dimension')
    parser.add_argument('--quant_dim', type=int, default=32,
                        help='Codebook embedding dimension')
    parser.add_argument('--codebook_size', type=int, default=8,
                        help='Number of action codes')
    parser.add_argument('--code_seq_len', type=int, default=4,
                        help='Number of action tokens per frame pair')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--spatial_depth', type=int, default=4,
                        help='Spatial transformer depth')
    parser.add_argument('--temporal_depth', type=int, default=4,
                        help='Temporal encoder depth')
    parser.add_argument('--predictor_depth', type=int, default=4,
                        help='Latent predictor depth')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dim_head', type=int, default=64,
                        help='Attention head dimension')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--num_train_steps', type=int, default=100005,
                        help='Total training steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--save_every', type=int, default=5000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--results_folder', type=str, default='results_concerto',
                        help='Results directory')
    
    # Feature arguments
    parser.add_argument('--freeze_concerto', action='store_true', default=True,
                        help='Freeze Concerto weights')
    parser.add_argument('--use_depth', action='store_true', default=False,
                        help='Use depth estimation for 3D lifting')
    parser.add_argument('--depth_model', type=str, default='dummy',
                        choices=['dummy', 'depth_anything', 'zoedepth'],
                        help='Depth estimation model')
    
    # WandB
    parser.add_argument('--wandb_project', type=str, default='concerto_laq',
                        help='WandB project name')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine if using pre-computed features
    use_precomputed = args.feature_cache_dir is not None
    
    # Feature size based on image size and Concerto's downsampling
    # Concerto typically uses grid_size=0.02, which gives roughly 1/8 downsampling
    feature_size = (args.image_size // 32, args.image_size // 32)  # Conservative estimate
    
    print(f"Creating ConcertoLAQ model...")
    print(f"  - Concerto model: {args.concerto_model}")
    print(f"  - Codebook size: {args.codebook_size}")
    print(f"  - Code sequence length: {args.code_seq_len}")
    print(f"  - Using pre-computed features: {use_precomputed}")
    
    # Create model
    model = ConcertoLAQ(
        concerto_model_name=args.concerto_model,
        concerto_dim=args.concerto_dim,
        dim=args.dim,
        quant_dim=args.quant_dim,
        codebook_size=args.codebook_size,
        code_seq_len=args.code_seq_len,
        image_size=args.image_size,
        feature_size=feature_size,
        spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth,
        predictor_depth=args.predictor_depth,
        dim_head=args.dim_head,
        heads=args.heads,
        freeze_concerto=args.freeze_concerto,
        use_depth=args.use_depth,
        depth_model_type=args.depth_model,
        use_precomputed_features=use_precomputed,
    ).cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = ConcertoLAQTrainer(
        model,
        folder=args.folder,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum_every=args.grad_accum,
        save_model_every=args.save_every,
        save_results_every=args.save_every,
        results_folder=args.results_folder,
        use_precomputed_features=use_precomputed,
        feature_cache_dir=args.feature_cache_dir,
        offset=args.offset,
        wandb_project=args.wandb_project,
    )
    
    print(f"\nStarting training...")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Total steps: {args.num_train_steps}")
    print(f"  - Results folder: {args.results_folder}")
    
    trainer.train()


if __name__ == '__main__':
    main()
