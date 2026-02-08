"""
Precompute VGGT+Concerto features for Concerto-LAQ training.

Runs the frozen ConcertoEncoder on all video pairs and saves the
SparseToDenseProjection output as .pt files. This allows training
the lightweight LAQ head without re-running the 1B+ parameter encoders.

Output: one .pt file per sample containing a [2, 14, 14, 512] float32 tensor.

Usage:
    # 4 GPUs
    torchrun --nproc_per_node=4 laq/precompute_features.py \
        --data_dir /mnt/nfs/eson/dataset/20bn-something-something-v2 \
        --output_dir /mnt/nfs/hanksome/precomputed_features

    # Test with small subset
    torchrun --nproc_per_node=4 laq/precompute_features.py \
        --data_dir /path/to/videos --output_dir /tmp/features --max_videos 100
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).parent))

from laq_model.webm_dataset import WebmVideoDataset
from laq_model.concerto_wrapper import ConcertoEncoder


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute VGGT+Concerto features")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .webm video files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save precomputed .pt features")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Limit number of videos (for testing)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--frame_offset", type=int, default=30)
    parser.add_argument("--samples_per_video", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--concerto_model", type=str, default="concerto_base")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    is_main = (rank == 0)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Precomputing features with {world_size} GPUs")
        print(f"Output directory: {args.output_dir}")

    # Synchronize so output_dir exists on all ranks
    if world_size > 1:
        dist.barrier()

    # Load encoder (frozen)
    encoder = ConcertoEncoder(
        concerto_model_name=args.concerto_model,
        freeze_concerto=True,
        device=device,
    )
    encoder.eval()

    # Dataset
    dataset = WebmVideoDataset(
        data_dir=args.data_dir,
        frame_size=(args.frame_size, args.frame_size),
        frame_offset=args.frame_offset,
        num_samples_per_video=args.samples_per_video,
        max_videos=args.max_videos,
    )

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if is_main:
        print(f"Total samples: {len(dataset)}")

    # Optional tqdm on rank 0
    iterator = dataloader
    if is_main:
        try:
            from tqdm import tqdm
            iterator = tqdm(dataloader, desc="Precomputing features", total=len(dataloader))
        except ImportError:
            pass

    processed = 0
    skipped = 0
    for batch in iterator:
        video = batch["video"].to(device)  # [B, C, 2, H, W]
        video_ids = batch["video_id"]

        # Skip samples that already have saved features
        to_process_idx = []
        for i, vid in enumerate(video_ids):
            out_path = os.path.join(args.output_dir, f"{vid}.pt")
            if not os.path.exists(out_path):
                to_process_idx.append(i)
            else:
                skipped += 1

        if len(to_process_idx) == 0:
            continue

        # Subset the batch to only unprocessed samples
        video_subset = video[to_process_idx]
        ids_subset = [video_ids[i] for i in to_process_idx]

        with torch.no_grad():
            features = encoder(video_subset)  # [B', 2, H', W', D]

        # Save each sample
        for i, vid in enumerate(ids_subset):
            out_path = os.path.join(args.output_dir, f"{vid}.pt")
            torch.save(features[i].cpu(), out_path)
            processed += 1

    if is_main:
        print(f"\nDone! Processed: {processed}, Skipped (existing): {skipped}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
