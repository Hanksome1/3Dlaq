"""
Pre-compute and cache Concerto features for faster training.

This script processes all video frames through the frozen Concerto encoder
and saves the features to disk. During training, these features can be
loaded directly instead of running Concerto on-the-fly.

Example usage:
    python precompute_concerto_features.py \
        --input_folder /path/to/videos \
        --output_folder /path/to/features \
        --concerto_model concerto_base \
        --batch_size 16
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import hashlib

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

from laq_model.concerto_wrapper import ConcertoEncoder, DepthEstimator


class VideoFrameDataset(Dataset):
    """Dataset that yields all frames from all videos."""
    
    def __init__(
        self,
        folder: str,
        image_size: int = 256,
    ):
        self.folder = folder
        self.image_size = image_size
        
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
        # Build index of all frames
        self.frames = []
        for video_folder in sorted(os.listdir(folder)):
            video_path = os.path.join(folder, video_folder)
            if os.path.isdir(video_path):
                frame_files = sorted(os.listdir(video_path))
                for i, frame_file in enumerate(frame_files):
                    frame_path = os.path.join(video_path, frame_file)
                    self.frames.append({
                        'path': frame_path,
                        'video_id': video_folder,
                        'frame_idx': i,
                    })
        
        print(f"Found {len(self.frames)} frames from {len(os.listdir(folder))} videos")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        info = self.frames[idx]
        img = Image.open(info['path'])
        img_tensor = self.transform(img)
        
        return {
            'image': img_tensor,
            'video_id': info['video_id'],
            'frame_idx': info['frame_idx'],
        }


def get_output_path(output_folder: str, video_id: str, frame_idx: int) -> str:
    """Generate output path for a feature file."""
    return os.path.join(output_folder, f"{video_id}_{frame_idx}.pt")


def feature_exists(output_folder: str, video_id: str, frame_idx: int) -> bool:
    """Check if feature file already exists."""
    return os.path.exists(get_output_path(output_folder, video_id, frame_idx))


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-compute Concerto features')
    
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to input video folder')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to output feature folder')
    parser.add_argument('--concerto_model', type=str, default='concerto_base',
                        choices=['concerto_small', 'concerto_base', 'concerto_large'],
                        help='Concerto model size')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for feature extraction')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--use_depth', action='store_true',
                        help='Use depth estimation')
    parser.add_argument('--depth_model', type=str, default='dummy',
                        choices=['dummy', 'depth_anything', 'zoedepth'],
                        help='Depth estimation model')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip frames with existing features')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Initialize Concerto encoder
    print(f"Loading Concerto model: {args.concerto_model}")
    depth_estimator = DepthEstimator(model_type=args.depth_model) if args.use_depth else None
    encoder = ConcertoEncoder(
        model_name=args.concerto_model,
        freeze=True,
        depth_estimator=depth_estimator,
    )
    encoder.eval()
    
    # Create dataset
    dataset = VideoFrameDataset(
        folder=args.input_folder,
        image_size=args.image_size,
    )
    
    # Filter out existing features if requested
    if args.skip_existing:
        original_len = len(dataset.frames)
        dataset.frames = [
            f for f in dataset.frames 
            if not feature_exists(args.output_folder, f['video_id'], f['frame_idx'])
        ]
        print(f"Skipping {original_len - len(dataset.frames)} existing features")
    
    if len(dataset) == 0:
        print("All features already computed. Exiting.")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Process frames
    print(f"Processing {len(dataset)} frames...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].cuda()  # [B, C, H, W]
            video_ids = batch['video_id']
            frame_idxs = batch['frame_idx']
            
            # Extract features
            features = encoder.encode_single_frame(images)  # [B, H', W', D]
            
            # Save each feature
            for i in range(len(video_ids)):
                video_id = video_ids[i]
                frame_idx = frame_idxs[i].item()
                feature = features[i].cpu()
                
                output_path = get_output_path(args.output_folder, video_id, frame_idx)
                torch.save(feature, output_path)
    
    print(f"Done! Features saved to {args.output_folder}")
    
    # Print summary
    num_files = len(os.listdir(args.output_folder))
    total_size = sum(
        os.path.getsize(os.path.join(args.output_folder, f))
        for f in os.listdir(args.output_folder)
    )
    print(f"Total files: {num_files}")
    print(f"Total size: {total_size / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
