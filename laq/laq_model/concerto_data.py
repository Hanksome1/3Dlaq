"""
Concerto-aware Dataset for Latent Action Quantization.

Extends the original ImageVideoDataset to support:
1. Depth estimation for 3D lifting
2. Pre-computed Concerto features loading
3. Camera parameter handling
"""

import os
import random
import hashlib
from pathlib import Path
from typing import Optional, Dict, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np


def exists(val):
    return val is not None


def pair(val):
    return val if isinstance(val, tuple) else (val, val)


class ConcertoVideoDataset(Dataset):
    """
    Dataset for Concerto-LAQ training.
    
    Supports three modes:
    1. Raw images with on-the-fly Concerto feature extraction
    2. Pre-computed Concerto features (fastest training)
    3. Raw images with depth estimation
    
    Args:
        folder: Path to video folder
        image_size: Target image size
        offset: Frame offset for frame pairs
        feature_cache_dir: Optional directory with pre-computed features
        use_depth: Whether to estimate depth maps
        depth_model_type: Type of depth estimator
        concerto_dim: Dimension of Concerto features (for pre-computed)
    """
    
    def __init__(
        self,
        folder: str,
        image_size: int = 256,
        offset: int = 5,
        feature_cache_dir: Optional[str] = None,
        use_depth: bool = False,
        depth_model_type: str = "dummy",
        concerto_dim: int = 256,
        feature_size: Tuple[int, int] = (8, 8),
    ):
        super().__init__()
        
        self.folder = folder
        self.image_size = pair(image_size)
        self.offset = offset
        self.use_precomputed = feature_cache_dir is not None
        self.feature_cache_dir = feature_cache_dir
        self.use_depth = use_depth
        self.concerto_dim = concerto_dim
        self.feature_size = feature_size
        
        # Get list of video folders
        self.folder_list = sorted(os.listdir(folder))
        
        # Cache folder lengths
        self._folder_lengths = {}
        
        # Image transforms
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(self.image_size),
            T.ToTensor(),
        ])
        
        # Depth estimator (lazy loaded)
        self._depth_estimator = None
        self.depth_model_type = depth_model_type
    
    @property
    def depth_estimator(self):
        """Lazy load depth estimator."""
        if self._depth_estimator is None and self.use_depth:
            from laq_model.concerto_wrapper import DepthEstimator
            self._depth_estimator = DepthEstimator(model_type=self.depth_model_type)
        return self._depth_estimator
    
    def _get_folder_length(self, folder_path: str) -> int:
        """Get cached folder length."""
        if folder_path not in self._folder_lengths:
            self._folder_lengths[folder_path] = len(os.listdir(folder_path))
        return self._folder_lengths[folder_path]
    
    def _get_cache_path(self, folder: str, frame_idx: int) -> str:
        """Get path for cached feature file."""
        if self.feature_cache_dir is None:
            return None
        
        # Create unique key
        key = f"{folder}_{frame_idx}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return os.path.join(self.feature_cache_dir, f"{hash_key}.pt")
    
    def _load_cached_features(self, folder: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Load pre-computed features if available."""
        cache_path = self._get_cache_path(folder, frame_idx)
        if cache_path and os.path.exists(cache_path):
            return torch.load(cache_path, map_location='cpu')
        return None
    
    def _get_frame_paths(self, folder: str, first_idx: int, second_idx: int) -> Tuple[str, str]:
        """Get paths to frame images."""
        folder_path = os.path.join(self.folder, folder)
        img_list = sorted(os.listdir(folder_path), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        
        first_path = os.path.join(folder_path, img_list[first_idx])
        second_path = os.path.join(folder_path, img_list[second_idx])
        
        return first_path, second_path
    
    def __len__(self):
        return len(self.folder_list)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dict with:
                - frames: [C, 2, H, W] video frames (if not using pre-computed)
                - features: [2, H', W', D] pre-computed features (if available)
                - depth: [2, H, W] depth maps (if use_depth)
        """
        try:
            folder = self.folder_list[index]
            folder_path = os.path.join(self.folder, folder)
            folder_len = self._get_folder_length(folder_path)
            
            # Select random frame pair
            first_idx = random.randint(0, max(0, folder_len - self.offset - 1))
            second_idx = min(first_idx + self.offset, folder_len - 1)
            
            result = {}
            
            if self.use_precomputed:
                # Try to load pre-computed features
                features_t0 = self._load_cached_features(folder, first_idx)
                features_t1 = self._load_cached_features(folder, second_idx)
                
                if features_t0 is not None and features_t1 is not None:
                    result['features'] = torch.stack([features_t0, features_t1], dim=0)
                    return result
            
            # Load raw images
            first_path, second_path = self._get_frame_paths(folder, first_idx, second_idx)
            
            img = Image.open(first_path)
            next_img = Image.open(second_path)
            
            frame_t0 = self.transform(img)
            frame_t1 = self.transform(next_img)
            
            # Stack frames: [C, 2, H, W]
            frames = torch.stack([frame_t0, frame_t1], dim=1)
            result['frames'] = frames
            
            # Estimate depth if needed
            if self.use_depth and self.depth_estimator is not None:
                with torch.no_grad():
                    frames_batch = frames.unsqueeze(0)  # [1, C, 2, H, W]
                    frames_for_depth = frames_batch.permute(0, 2, 1, 3, 4)  # [1, 2, C, H, W]
                    depth = self.depth_estimator(frames_for_depth)  # [1, 2, H, W]
                    result['depth'] = depth.squeeze(0)  # [2, H, W]
            
            return result
            
        except Exception as e:
            print(f"Error loading sample {index}: {e}")
            # Return next sample on error
            if index < len(self) - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, len(self) - 1))


class PrecomputedFeatureDataset(Dataset):
    """
    Dataset that loads only pre-computed Concerto features.
    
    Much faster for training since no image loading or feature extraction needed.
    
    Args:
        feature_dir: Directory containing .pt feature files
        offset: Frame offset between feature pairs
    """
    
    def __init__(
        self,
        feature_dir: str,
        offset: int = 5,
        feature_dim: int = 256,
    ):
        super().__init__()
        
        self.feature_dir = feature_dir
        self.offset = offset
        self.feature_dim = feature_dim
        
        # Index all feature files
        self._build_index()
    
    def _build_index(self):
        """Build index of features organized by video."""
        self.videos = {}
        
        for file in os.listdir(self.feature_dir):
            if file.endswith('.pt'):
                # Parse filename: {video_id}_{frame_idx}.pt
                parts = file[:-3].rsplit('_', 1)
                if len(parts) == 2:
                    video_id, frame_idx = parts
                    frame_idx = int(frame_idx)
                    
                    if video_id not in self.videos:
                        self.videos[video_id] = []
                    self.videos[video_id].append(frame_idx)
        
        # Sort frame indices
        for video_id in self.videos:
            self.videos[video_id] = sorted(self.videos[video_id])
        
        # Create flat list of valid pairs
        self.pairs = []
        for video_id, frames in self.videos.items():
            for i, frame_idx in enumerate(frames[:-1]):
                # Find frame with offset
                target_idx = frame_idx + self.offset
                if target_idx in frames:
                    self.pairs.append((video_id, frame_idx, target_idx))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        video_id, frame_t0, frame_t1 = self.pairs[index]
        
        path_t0 = os.path.join(self.feature_dir, f"{video_id}_{frame_t0}.pt")
        path_t1 = os.path.join(self.feature_dir, f"{video_id}_{frame_t1}.pt")
        
        features_t0 = torch.load(path_t0, map_location='cpu')
        features_t1 = torch.load(path_t1, map_location='cpu')
        
        return {
            'features': torch.stack([features_t0, features_t1], dim=0),
        }


def collate_fn(batch):
    """
    Custom collate function to handle mixed content.
    
    Handles batches with either:
    - Raw frames
    - Pre-computed features
    - Both frames and depth
    """
    result = {}
    
    # Check what fields are present
    sample = batch[0]
    
    if 'frames' in sample:
        result['frames'] = torch.stack([s['frames'] for s in batch], dim=0)
    
    if 'features' in sample:
        result['features'] = torch.stack([s['features'] for s in batch], dim=0)
    
    if 'depth' in sample:
        result['depth'] = torch.stack([s['depth'] for s in batch], dim=0)
    
    return result
