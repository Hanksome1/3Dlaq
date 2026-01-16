"""
Concerto Wrapper Module for Latent Action Quantization.

This module provides a wrapper around the pre-trained Concerto model (PTv3 backbone)
for extracting 2D+3D aware features from video frames.

Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations
https://github.com/Pointcept/Concerto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, Union


class DepthEstimator(nn.Module):
    """
    Monocular depth estimation wrapper.
    
    Supports multiple backends:
    - depth_anything: Fast and accurate monocular depth
    - zoedepth: High quality depth estimation
    - dummy: Returns uniform depth for testing
    """
    
    def __init__(
        self,
        model_type: str = "depth_anything",
        model_size: str = "base",
        device: str = "cuda",
    ):
        super().__init__()
        self.model_type = model_type
        self.device = device
        self.model = None
        
        if model_type == "depth_anything":
            self._load_depth_anything(model_size)
        elif model_type == "zoedepth":
            self._load_zoedepth()
        elif model_type == "dummy":
            pass  # No model needed
        else:
            raise ValueError(f"Unknown depth model type: {model_type}")
    
    def _load_depth_anything(self, model_size: str = "base"):
        """Load Depth Anything model."""
        try:
            from transformers import pipeline
            self.model = pipeline(
                task="depth-estimation",
                model=f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf",
                device=0 if self.device == "cuda" else -1,
            )
        except ImportError:
            print("Warning: transformers not installed. Using dummy depth estimator.")
            self.model_type = "dummy"
    
    def _load_zoedepth(self):
        """Load ZoeDepth model."""
        try:
            import torch.hub
            self.model = torch.hub.load(
                "isl-org/ZoeDepth", "ZoeD_NK", pretrained=True
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Failed to load ZoeDepth: {e}. Using dummy depth estimator.")
            self.model_type = "dummy"
    
    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,  # [B, C, H, W] or [B, T, C, H, W]
    ) -> torch.Tensor:
        """
        Estimate depth from RGB images.
        
        Args:
            images: RGB images in [0, 1] range
            
        Returns:
            depth: Depth maps [B, H, W] or [B, T, H, W]
        """
        has_time_dim = images.ndim == 5
        if has_time_dim:
            B, T, C, H, W = images.shape
            images = images.reshape(B * T, C, H, W)
        
        if self.model_type == "dummy":
            # Return uniform depth
            depth = torch.ones(images.shape[0], images.shape[2], images.shape[3], device=images.device)
        elif self.model_type == "depth_anything":
            # Process through pipeline (expects PIL images or numpy)
            depth_list = []
            for img in images:
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                result = self.model(img_np)
                depth_list.append(torch.from_numpy(np.array(result["depth"])))
            depth = torch.stack(depth_list).to(images.device)
        elif self.model_type == "zoedepth":
            depth = self.model.infer(images)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        if has_time_dim:
            depth = depth.reshape(B, T, H, W)
        
        return depth


class PointCloudLifter(nn.Module):
    """
    Lifts 2D images to 3D point clouds using depth maps.
    
    Handles back-projection from 2D to 3D using camera intrinsics.
    """
    
    def __init__(
        self,
        default_fx: float = 500.0,  # Default focal length (pixels)
        default_fy: float = 500.0,
        default_cx: Optional[float] = None,  # Principal point (default: image center)
        default_cy: Optional[float] = None,
    ):
        super().__init__()
        self.default_fx = default_fx
        self.default_fy = default_fy
        self.default_cx = default_cx
        self.default_cy = default_cy
    
    def forward(
        self,
        rgb: torch.Tensor,  # [B, C, H, W]
        depth: torch.Tensor,  # [B, H, W]
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Lift 2D image to 3D point cloud.
        
        Args:
            rgb: RGB image [B, C, H, W]
            depth: Depth map [B, H, W]
            fx, fy: Focal lengths
            cx, cy: Principal points
            
        Returns:
            dict with:
                coord: [B, N, 3] point coordinates
                color: [B, N, 3] point colors (normalized)
                batch: [B, N] batch indices (for batched processing)
        """
        B, C, H, W = rgb.shape
        device = rgb.device
        
        # Use defaults if not provided
        fx = fx or self.default_fx
        fy = fy or self.default_fy
        cx = cx if cx is not None else W / 2.0
        cy = cy if cy is not None else H / 2.0
        
        # Create pixel coordinate grids
        u = torch.arange(W, device=device).float()
        v = torch.arange(H, device=device).float()
        u, v = torch.meshgrid(u, v, indexing='xy')
        u = u.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        v = v.unsqueeze(0).expand(B, -1, -1)
        
        # Back-project to 3D
        z = depth  # [B, H, W]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack coordinates
        coords = torch.stack([x, y, z], dim=-1)  # [B, H, W, 3]
        coords = coords.reshape(B, H * W, 3)  # [B, N, 3]
        
        # Get colors (permute to [B, H, W, C] then flatten)
        colors = rgb.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, 3]
        
        # Create batch indices for Concerto's batch format
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, H * W)
        batch_indices = batch_indices.reshape(B * H * W)  # [B*N]
        
        return {
            "coord": coords,
            "color": colors,
            "batch": batch_indices,
            "height": H,
            "width": W,
        }


class ConcertoEncoder(nn.Module):
    """
    Wrapper for Concerto pre-trained model.
    
    Provides interface for extracting 2D+3D aware features from video frames.
    The features can be used for downstream tasks like latent action prediction.
    
    Args:
        model_name: One of "concerto_small" (39M), "concerto_base" (108M), "concerto_large" (208M)
        freeze: Whether to freeze Concerto weights
        device: Device to run on
        depth_estimator: Optional depth estimator instance
        grid_size: Grid size for point cloud sampling (controls point density)
    """
    
    # Output dimensions for each model size
    MODEL_DIMS = {
        "concerto_small": 256,
        "concerto_base": 256,
        "concerto_large": 512,
    }
    
    def __init__(
        self,
        model_name: str = "concerto_base",
        freeze: bool = True,
        device: str = "cuda",
        depth_estimator: Optional[DepthEstimator] = None,
        grid_size: float = 0.02,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.freeze = freeze
        self.grid_size = grid_size
        self.output_dim = self.MODEL_DIMS.get(model_name, 256)
        
        # Load Concerto model
        self.model = self._load_model(model_name)
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # Depth estimator
        self.depth_estimator = depth_estimator or DepthEstimator(model_type="dummy")
        
        # Point cloud lifter
        self.lifter = PointCloudLifter()
        
        # Transform pipeline (following Concerto's default)
        self.transform = None
        self._setup_transform()
    
    def _load_model(self, model_name: str):
        """Load pre-trained Concerto model."""
        try:
            import concerto
            model = concerto.model.load(
                model_name,
                repo_id="Pointcept/Concerto"
            ).to(self.device)
            print(f"Loaded Concerto model: {model_name}")
            return model
        except ImportError:
            print("Warning: concerto package not installed. Using dummy encoder.")
            return None
        except Exception as e:
            print(f"Warning: Failed to load Concerto model: {e}. Using dummy encoder.")
            return None
    
    def _setup_transform(self):
        """Set up data transform pipeline following Concerto's defaults."""
        try:
            import concerto
            self.transform = concerto.transform.default()
        except ImportError:
            self.transform = None
    
    def _apply_transform(self, point_dict: Dict) -> Dict:
        """Apply Concerto's transform pipeline to point data."""
        if self.transform is not None:
            return self.transform(point_dict)
        
        # Simple fallback transform
        point_dict["coord"] = point_dict["coord"].float()
        point_dict["color"] = point_dict["color"].float()
        return point_dict
    
    @torch.no_grad()
    def encode_single_frame(
        self,
        rgb: torch.Tensor,  # [B, C, H, W]
        depth: Optional[torch.Tensor] = None,  # [B, H, W]
    ) -> torch.Tensor:
        """
        Encode a single frame to Concerto features.
        
        Args:
            rgb: RGB image [B, C, H, W] in [0, 1] range
            depth: Optional depth map [B, H, W]
            
        Returns:
            features: [B, H, W, D] feature map in image space
        """
        B, C, H, W = rgb.shape
        
        # Estimate depth if not provided
        if depth is None:
            depth = self.depth_estimator(rgb)
        
        # Lift to 3D point cloud
        point_data = self.lifter(rgb, depth)
        
        if self.model is None:
            # Return dummy features if model not loaded
            return torch.randn(B, H, W, self.output_dim, device=rgb.device)
        
        # Process each batch item through Concerto
        all_features = []
        for b in range(B):
            # Prepare single point cloud
            single_point = {
                "coord": point_data["coord"][b].cpu().numpy(),
                "color": point_data["color"][b].cpu().numpy(),
            }
            
            # Apply transform
            single_point = self._apply_transform(single_point)
            
            # Move to device and add batch dimension handling
            for key in single_point:
                if isinstance(single_point[key], np.ndarray):
                    single_point[key] = torch.from_numpy(single_point[key]).to(self.device)
            
            # Run through Concerto
            with torch.no_grad():
                output = self.model(single_point)
            
            # Get features and reshape to image space
            # Concerto outputs per-point features that we project back to 2D
            point_features = output.get("feat", output)  # [N, D]
            if isinstance(point_features, torch.Tensor):
                # Reshape to image space
                features_2d = point_features.reshape(H, W, -1)
                all_features.append(features_2d)
            else:
                all_features.append(torch.randn(H, W, self.output_dim, device=self.device))
        
        return torch.stack(all_features, dim=0)  # [B, H, W, D]
    
    def encode_video_pair(
        self,
        video: torch.Tensor,  # [B, C, 2, H, W]
        depth: Optional[torch.Tensor] = None,  # [B, 2, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a pair of video frames.
        
        Args:
            video: Two consecutive frames [B, C, 2, H, W]
            depth: Optional depth maps [B, 2, H, W]
            
        Returns:
            features_t0: Features of first frame [B, H, W, D]
            features_t1: Features of second frame [B, H, W, D]
        """
        B, C, T, H, W = video.shape
        assert T == 2, "Expected exactly 2 frames"
        
        frame_t0 = video[:, :, 0]  # [B, C, H, W]
        frame_t1 = video[:, :, 1]
        
        depth_t0 = depth[:, 0] if depth is not None else None
        depth_t1 = depth[:, 1] if depth is not None else None
        
        features_t0 = self.encode_single_frame(frame_t0, depth_t0)
        features_t1 = self.encode_single_frame(frame_t1, depth_t1)
        
        return features_t0, features_t1
    
    def forward(
        self,
        video: torch.Tensor,  # [B, C, 2, H, W]
        depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass encoding video frames.
        
        Args:
            video: Two consecutive frames [B, C, 2, H, W]
            depth: Optional depth maps [B, 2, H, W]
            
        Returns:
            features: Stacked features [B, 2, H, W, D]
        """
        features_t0, features_t1 = self.encode_video_pair(video, depth)
        return torch.stack([features_t0, features_t1], dim=1)  # [B, 2, H, W, D]


class ConcertoFeatureCache:
    """
    Cache for pre-computed Concerto features.
    
    Useful for datasets where we want to pre-compute features offline
    and load them during training for faster iteration.
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, video_path: str, frame_idx: int) -> str:
        """Get cache file path for a specific frame."""
        import os
        import hashlib
        
        # Create unique key from video path and frame index
        key = f"{video_path}_{frame_idx}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{hash_key}.pt")
    
    def has_cached(self, video_path: str, frame_idx: int) -> bool:
        """Check if features are cached."""
        import os
        return os.path.exists(self.get_cache_path(video_path, frame_idx))
    
    def save(self, features: torch.Tensor, video_path: str, frame_idx: int):
        """Save features to cache."""
        cache_path = self.get_cache_path(video_path, frame_idx)
        torch.save(features.cpu(), cache_path)
    
    def load(self, video_path: str, frame_idx: int, device: str = "cuda") -> torch.Tensor:
        """Load features from cache."""
        cache_path = self.get_cache_path(video_path, frame_idx)
        return torch.load(cache_path, map_location=device)
