"""
Dataset classes for Something-Something v2 webm videos.

Supports:
1. WebmVideoDataset - Load video pairs directly from .webm files
2. Precomputed features for faster training

Dataset structure expected:
    /mnt/nfs/eson/dataset/20bn-something-something-v2/
        ├── 1.webm
        ├── 2.webm
        ├── ...
        └── 220847.webm
"""

import os
import glob
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WebmVideoDataset(Dataset):
    """
    Dataset for loading frame pairs from .webm video files.
    
    Each sample returns two consecutive frames for action learning.
    
    Args:
        data_dir: Directory containing .webm files
        frame_size: Target frame size (H, W), must be divisible by 14 for VGGT
        frame_offset: Target number of frames between the pair (default: 30)
        min_frame_offset: Minimum acceptable frame offset (default: 5)
        num_samples_per_video: Number of pairs to sample per video (None = all)
        max_videos: Maximum number of videos to use (None = all)
        transform: Optional transform to apply to frames
    """
    
    def __init__(
        self,
        data_dir: str,
        frame_size: Tuple[int, int] = (224, 224),
        frame_offset: int = 30,
        min_frame_offset: int = 5,
        num_samples_per_video: Optional[int] = 1,
        max_videos: Optional[int] = None,
        transform = None,
    ):
        self.data_dir = data_dir
        self.frame_size = frame_size
        self.frame_offset = frame_offset
        self.min_frame_offset = min_frame_offset
        self.num_samples_per_video = num_samples_per_video
        self.transform = transform
        
        # Track bad videos to avoid repeated warnings
        self._bad_videos = set()
        self._warned_videos = set()
        
        # Find all webm files
        self.video_files = sorted(glob.glob(os.path.join(data_dir, "*.webm")))
        
        if max_videos is not None:
            self.video_files = self.video_files[:max_videos]
        
        print(f"Found {len(self.video_files)} video files in {data_dir}")
        
        # Build index: list of (video_path, start_frame_idx, actual_offset)
        self.samples = self._build_sample_index()
        print(f"Total samples: {len(self.samples)}")
    
    def _build_sample_index(self) -> List[Tuple[str, int, int]]:
        """Build list of (video_path, start_frame, offset) tuples with adaptive offset."""
        samples = []
        
        import cv2
        
        skipped_count = 0
        for video_path in self.video_files:
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    self._bad_videos.add(video_path)
                    skipped_count += 1
                    continue
                    
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Skip videos that are too short even for min_frame_offset
                if frame_count <= self.min_frame_offset:
                    self._bad_videos.add(video_path)
                    skipped_count += 1
                    continue
                
                # Use adaptive offset: prefer frame_offset, but use smaller if needed
                actual_offset = min(self.frame_offset, frame_count - 2)
                actual_offset = max(actual_offset, self.min_frame_offset)
                
                # Valid start frames
                max_start = frame_count - actual_offset - 1
                
                if max_start < 0:
                    self._bad_videos.add(video_path)
                    skipped_count += 1
                    continue
                
                if self.num_samples_per_video is None:
                    # Use all possible pairs
                    for start in range(max_start + 1):
                        samples.append((video_path, start, actual_offset))
                else:
                    # Sample uniformly
                    if max_start >= 0:
                        step = max(1, (max_start + 1) // self.num_samples_per_video)
                        for i in range(min(self.num_samples_per_video, max_start + 1)):
                            start = min(i * step, max_start)
                            samples.append((video_path, start, actual_offset))
            except Exception as e:
                self._bad_videos.add(video_path)
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} videos (too short or corrupted)")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a frame pair.
        
        Returns:
            Dict with:
                - video: [C, 2, H, W] tensor of two frames
                - video_id: video filename (for logging)
        """
        import cv2
        
        video_path, start_frame, offset = self.samples[idx]
        
        # If this video is known to be bad, try another
        if video_path in self._bad_videos:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("Cannot open video")
            
            # Read first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret1, frame1 = cap.read()
            
            # Read second frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + offset)
            ret2, frame2 = cap.read()
            
            cap.release()
            
            if not ret1 or not ret2 or frame1 is None or frame2 is None:
                raise ValueError("Cannot read frames")
            
            # Convert BGR to RGB and resize
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            
            frame1 = cv2.resize(frame1, self.frame_size[::-1])  # cv2 uses (W, H)
            frame2 = cv2.resize(frame2, self.frame_size[::-1])
            
            # Convert to tensor [C, H, W] in [0, 1]
            frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float() / 255.0
            frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float() / 255.0
            
            # Stack as [C, 2, H, W]
            video = torch.stack([frame1, frame2], dim=1)
            
            if self.transform is not None:
                video = self.transform(video)
            
            video_id = Path(video_path).stem
            
            return {
                "video": video,
                "video_id": video_id,
            }
            
        except Exception as e:
            # Mark as bad and warn only once
            self._bad_videos.add(video_path)
            if video_path not in self._warned_videos:
                print(f"Warning: Failed to read frames from {video_path}")
                self._warned_videos.add(video_path)
            
            # Try another random sample
            return self.__getitem__(random.randint(0, len(self) - 1))


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for WebmVideoDataset."""
    videos = torch.stack([item["video"] for item in batch], dim=0)
    video_ids = [item["video_id"] for item in batch]
    
    return {
        "video": videos,
        "video_ids": video_ids,
    }


def create_dataloader(
    data_dir: str,
    batch_size: int = 4,
    frame_size: Tuple[int, int] = (224, 224),
    frame_offset: int = 30,
    min_frame_offset: int = 5,
    num_samples_per_video: int = 1,
    max_videos: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for the webm video dataset."""
    
    dataset = WebmVideoDataset(
        data_dir=data_dir,
        frame_size=frame_size,
        frame_offset=frame_offset,
        min_frame_offset=min_frame_offset,
        num_samples_per_video=num_samples_per_video,
        max_videos=max_videos,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    # Test the dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, 
                        default="/mnt/nfs/eson/dataset/20bn-something-something-v2")
    parser.add_argument("--max_videos", type=int, default=10)
    args = parser.parse_args()
    
    print("Testing WebmVideoDataset...")
    
    dataset = WebmVideoDataset(
        data_dir=args.data_dir,
        frame_size=(224, 224),
        max_videos=args.max_videos,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample video shape: {sample['video'].shape}")
    print(f"Sample video_id: {sample['video_id']}")
    
    # Test dataloader
    loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=2,
        max_videos=args.max_videos,
    )
    
    batch = next(iter(loader))
    print(f"\nBatch video shape: {batch['video'].shape}")
    print(f"Batch video_ids: {batch['video_ids']}")
    
    print("\n✓ Dataset test passed!")
