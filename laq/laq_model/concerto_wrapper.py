"""
Concerto Wrapper Module with VGGT Integration.

This module provides a wrapper around the pre-trained Concerto model (PTv3 backbone)
using VGGT (Visual Geometry Grounded Transformer) for:
- Single-frame depth estimation
- Camera intrinsics estimation  
- 3D point cloud generation

VGGT: https://github.com/facebookresearch/vggt
Concerto: https://github.com/Pointcept/Concerto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, Union, List
from pathlib import Path


class VGGTEncoder(nn.Module):
    """
    VGGT-based encoder for single-frame depth and camera estimation.
    
    VGGT can estimate from a single image:
    - Depth map
    - Camera intrinsics (focal length, principal point)
    - Camera extrinsics (pose)
    - Direct point map
    
    Args:
        model_name: VGGT model variant (default: "facebook/VGGT-1B")
        device: Device to run on
        dtype: Data type for inference (bfloat16 on Ampere+, else float16)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/VGGT-1B",
        device: str = "cuda",
        use_bfloat16: bool = True,
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.model = None
        
        # Determine dtype based on GPU capability
        if device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if (capability >= 8 and use_bfloat16) else torch.float16
        else:
            self.dtype = torch.float32
        
        self._load_model()
    
    def _load_model(self):
        """Load VGGT model from HuggingFace."""
        try:
            from vggt.models.vggt import VGGT
            self.model = VGGT.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print(f"Loaded VGGT model: {self.model_name}")
        except ImportError:
            print("Warning: vggt package not installed. Using dummy encoder.")
            print("Install with: pip install -e . (from vggt repo)")
            self.model = None
        except Exception as e:
            print(f"Warning: Failed to load VGGT model: {e}")
            self.model = None
    
    @staticmethod
    def preprocess_images(images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for VGGT.
        
        Args:
            images: [B, C, H, W] in [0, 1] range
            
        Returns:
            Preprocessed images ready for VGGT
        """
        # VGGT expects images in a specific format
        # Normalize if needed (VGGT's load_and_preprocess_images handles this)
        return images
    
    @torch.no_grad()
    def encode_single_frame(
        self,
        image: torch.Tensor,  # [B, C, H, W] or [C, H, W]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a single frame to get depth, camera params, and point cloud.
        
        Args:
            image: RGB image in [0, 1] range
            
        Returns:
            Dict with:
                - depth_map: [B, H, W] depth values
                - intrinsic: [B, 3, 3] camera intrinsic matrix
                - extrinsic: [B, 4, 4] camera extrinsic matrix
                - point_map: [B, H, W, 3] 3D point coordinates
                - point_conf: [B, H, W] confidence scores
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        if self.model is None:
            # Return dummy values for testing
            return {
                'depth_map': torch.ones(B, H, W, device=self.device),
                'intrinsic': torch.eye(3, device=self.device).unsqueeze(0).expand(B, -1, -1),
                'extrinsic': torch.eye(4, device=self.device).unsqueeze(0).expand(B, -1, -1),
                'point_map': self._dummy_point_map(B, H, W),
                'point_conf': torch.ones(B, H, W, device=self.device),
            }
        
        # Add batch dimension for VGGT (it expects [batch, num_images, C, H, W])
        images = image.unsqueeze(0)  # [1, B, C, H, W] - treating B as num_images
        
        with torch.cuda.amp.autocast(dtype=self.dtype):
            # Get aggregated tokens
            aggregated_tokens_list, ps_idx = self.model.aggregator(images)
            
            # Predict cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            
            # Predict depth
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
            
            # Predict point map (alternative to unprojection)
            point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)
        
        # Remove the extra batch dimension
        return {
            'depth_map': depth_map.squeeze(0),  # [B, H, W]
            'intrinsic': intrinsic.squeeze(0),  # [B, 3, 3]
            'extrinsic': extrinsic.squeeze(0),  # [B, 4, 4]
            'point_map': point_map.squeeze(0),  # [B, H, W, 3]
            'point_conf': point_conf.squeeze(0),  # [B, H, W]
        }
    
    def _dummy_point_map(self, B: int, H: int, W: int) -> torch.Tensor:
        """Generate dummy point map for testing without VGGT."""
        # Create a simple grid-based point cloud
        u = torch.linspace(-1, 1, W, device=self.device)
        v = torch.linspace(-1, 1, H, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        z = torch.ones_like(uu)
        
        point_map = torch.stack([uu, vv, z], dim=-1)  # [H, W, 3]
        return point_map.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 3]
    
    @torch.no_grad()
    def encode_video_pair(
        self,
        video: torch.Tensor,  # [B, C, 2, H, W]
    ) -> Tuple[Dict, Dict]:
        """
        Encode a pair of video frames.
        
        Args:
            video: Two consecutive frames [B, C, 2, H, W]
            
        Returns:
            results_t0: Dict with depth, intrinsic, extrinsic, point_map for frame 0
            results_t1: Dict with depth, intrinsic, extrinsic, point_map for frame 1
        """
        frame_t0 = video[:, :, 0]  # [B, C, H, W]
        frame_t1 = video[:, :, 1]  # [B, C, H, W]
        
        results_t0 = self.encode_single_frame(frame_t0)
        results_t1 = self.encode_single_frame(frame_t1)
        
        return results_t0, results_t1


class ConcertoEncoder(nn.Module):
    """
    Wrapper for Concerto pre-trained model with VGGT integration.
    
    Pipeline:
    1. VGGT extracts depth + camera params + point cloud from each frame
    2. Point cloud is processed through Concerto (PTv3)
    3. Features are mapped back to 2D for downstream tasks
    
    Args:
        concerto_model_name: Concerto model size
        vggt_model_name: VGGT model name on HuggingFace
        freeze_concerto: Whether to freeze Concerto weights
        freeze_vggt: Whether to freeze VGGT weights (should be True)
        device: Device to run on
    """
    
    MODEL_DIMS = {
        "concerto_small": 256,
        "concerto_base": 256,
        "concerto_large": 512,
    }
    
    def __init__(
        self,
        concerto_model_name: str = "concerto_base",
        vggt_model_name: str = "facebook/VGGT-1B",
        freeze_concerto: bool = True,
        freeze_vggt: bool = True,
        device: str = "cuda",
        grid_size: float = 0.02,
    ):
        super().__init__()
        self.device = device
        self.concerto_model_name = concerto_model_name
        self.output_dim = self.MODEL_DIMS.get(concerto_model_name, 256)
        self.grid_size = grid_size
        
        # VGGT encoder for depth + camera estimation
        self.vggt = VGGTEncoder(
            model_name=vggt_model_name,
            device=device,
        )
        if freeze_vggt and self.vggt.model is not None:
            for param in self.vggt.parameters():
                param.requires_grad = False
        
        # Concerto encoder
        self.concerto = self._load_concerto(concerto_model_name)
        if freeze_concerto and self.concerto is not None:
            for param in self.concerto.parameters():
                param.requires_grad = False
            self.concerto.eval()
        
        # Transform pipeline for Concerto
        self.transform = None
        self._setup_transform()
    
    def _load_concerto(self, model_name: str):
        """Load pre-trained Concerto model."""
        # Try multiple import approaches
        try:
            import concerto
            print(f"Concerto package found at: {concerto.__file__}")
            
            # Try with flash attention first
            try:
                model = concerto.model.load(
                    model_name,
                    repo_id="Pointcept/Concerto"
                ).to(self.device)
                print(f"Loaded Concerto model: {model_name}")
                return model
            except AssertionError as e:
                if "flash_attn" in str(e):
                    print("Flash Attention not available, loading with enable_flash=False...")
                    # Load without flash attention
                    custom_config = dict(
                        enc_patch_size=[1024 for _ in range(5)],
                        enable_flash=False,
                    )
                    model = concerto.model.load(
                        model_name,
                        repo_id="Pointcept/Concerto",
                        custom_config=custom_config
                    ).to(self.device)
                    print(f"Loaded Concerto model (no flash): {model_name}")
                    return model
                raise e
                
        except ImportError as e:
            print(f"Warning: concerto package not installed. Error: {e}")
            print("Using dummy encoder.")
            return None
        except Exception as e:
            print(f"Warning: Failed to load Concerto model: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _setup_transform(self):
        """Set up Concerto's data transform pipeline."""
        try:
            import concerto
            self.transform = concerto.transform.default()
        except ImportError:
            self.transform = None
    
    def _fps_subsample(self, points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Farthest Point Sampling (FPS) for point cloud downsampling.
        
        Iteratively selects points that are farthest from already selected points,
        ensuring good spatial coverage.
        
        Args:
            points: [N, 3] point coordinates
            num_samples: number of points to sample
            
        Returns:
            indices: [num_samples] indices of selected points
        """
        N = points.shape[0]
        
        # Try to use PyTorch3D's FPS (faster GPU implementation)
        try:
            from pytorch3d.ops import sample_farthest_points
            # pytorch3d expects [B, N, 3]
            points_batch = points.unsqueeze(0)
            _, indices = sample_farthest_points(points_batch, K=num_samples)
            return indices.squeeze(0)
        except ImportError:
            pass
        
        # Try to use torch_cluster's FPS
        try:
            from torch_cluster import fps
            batch = torch.zeros(N, dtype=torch.long)
            ratio = num_samples / N
            indices = fps(points, batch, ratio=ratio)[:num_samples]
            return indices
        except ImportError:
            pass
        
        # Fallback: Simple CPU implementation
        device = points.device
        points = points.cpu().numpy()
        
        indices = np.zeros(num_samples, dtype=np.int64)
        distances = np.full(N, np.inf)
        
        # Start with random point
        indices[0] = np.random.randint(N)
        
        for i in range(1, num_samples):
            last_point = points[indices[i-1]]
            # Update distances
            dist_to_last = np.sum((points - last_point) ** 2, axis=1)
            distances = np.minimum(distances, dist_to_last)
            # Select farthest point
            indices[i] = np.argmax(distances)
        
        return torch.from_numpy(indices).to(device)
    
    def point_map_to_concerto_input(
        self,
        point_map: torch.Tensor,  # [B, H, W, 3]
        colors: torch.Tensor,  # [B, C, H, W]
        max_points: int = 8192,  # Concerto works best with fewer points
    ) -> List[Dict]:
        """
        Convert VGGT point map to Concerto input format.
        
        Concerto expects a dict with:
        - coord: [N, 3] point coordinates
        - color: [N, 3] normalized colors (0-1)
        - normal: [N, 3] surface normals (we use dummy zeros)
        - grid_coord: [N, 3] grid coordinates (for sparse conv)
        - feat: [N, 9] input features (coord + color + normal)
        
        Args:
            point_map: 3D coordinates [B, H, W, 3]
            colors: RGB values [B, C, H, W]
            max_points: Maximum number of points (subsample if exceeded)
            
        Returns:
            List of point dictionaries for each batch item
        """
        B, H, W, _ = point_map.shape
        
        # Reshape colors to match point_map: [B, H, W, C]
        colors_hwc = colors.permute(0, 2, 3, 1)
        
        batch_points = []
        for b in range(B):
            # Flatten spatial dimensions
            coord = point_map[b].reshape(-1, 3).cpu().float()  # [N, 3]
            color = colors_hwc[b].reshape(-1, 3).cpu().float()  # [N, 3]
            
            N = coord.shape[0]
            
            # Subsample if too many points using Farthest Point Sampling (FPS)
            if N > max_points:
                indices = self._fps_subsample(coord, max_points)
                coord = coord[indices]
                color = color[indices]
            
            # Create dummy normals (zeros) since VGGT doesn't provide them
            normal = torch.zeros_like(coord)  # [N, 3]
            
            # Center the point cloud
            coord = coord - coord.mean(dim=0)
            
            # Grid sampling (voxelization)
            grid_coord = torch.floor(coord / self.grid_size).int()
            
            # Prepare the dict in Concerto's expected format
            # feat should be 9-dim: coord (3) + color (3) + normal (3)
            point_dict = {
                "coord": coord,  # [N, 3] float tensor
                "grid_coord": grid_coord,  # [N, 3] int tensor
                "color": color,  # [N, 3] float tensor  
                "normal": normal,  # [N, 3] float tensor
                "feat": torch.cat([coord, color, normal], dim=1),  # [N, 9] combined features
                "offset": torch.tensor([coord.shape[0]], dtype=torch.int64),  # batch offset
            }
            
            batch_points.append(point_dict)
        
        return batch_points
    
    @torch.no_grad()
    def encode_single_frame(
        self,
        image: torch.Tensor,  # [B, C, H, W]
    ) -> torch.Tensor:
        """
        Encode a single frame to Concerto features.
        
        Args:
            image: RGB image [B, C, H, W] in [0, 1] range
            
        Returns:
            features: [B, H', W', D] Concerto features (downsampled from input resolution)
        """
        B, C, H, W = image.shape
        
        # Target feature size (to avoid OOM in attention)
        # Concerto typically operates at grid_size=0.02, giving ~16x16 for 256x256 input
        target_h = max(8, H // 16)
        target_w = max(8, W // 16)
        
        # Step 1: Get point cloud from VGGT
        vggt_results = self.vggt.encode_single_frame(image)
        point_map = vggt_results['point_map']  # [B, H, W, 3]
        
        if self.concerto is None:
            # Return dummy features at reduced resolution
            dummy_features = torch.randn(B, H, W, self.output_dim, device=self.device)
            # Downsample to target size
            dummy_features = dummy_features.permute(0, 3, 1, 2)  # [B, D, H, W]
            dummy_features = F.interpolate(dummy_features, size=(target_h, target_w), mode='bilinear', align_corners=False)
            dummy_features = dummy_features.permute(0, 2, 3, 1)  # [B, H', W', D]
            return dummy_features
        
        # Step 2: Prepare Concerto input
        batch_points = self.point_map_to_concerto_input(point_map, image)
        
        # Step 3: Run through Concerto
        all_features = []
        for point_dict in batch_points:
            # Move all tensors to device
            for key in point_dict:
                if isinstance(point_dict[key], torch.Tensor):
                    point_dict[key] = point_dict[key].to(self.device)
                elif isinstance(point_dict[key], np.ndarray):
                    point_dict[key] = torch.from_numpy(point_dict[key]).to(self.device)
            
            # Run Concerto
            try:
                output = self.concerto(point_dict)
            except Exception as e:
                print(f"Concerto forward error: {e}")
                import traceback
                traceback.print_exc()
                # Return dummy on error
                feat = torch.randn(target_h, target_w, self.output_dim, device=self.device)
                all_features.append(feat)
                continue
            
            # Get features
            if isinstance(output, dict):
                feat = output.get("feat", output.get("features"))
            else:
                feat = output
            
            # Reshape to image space and downsample
            if feat is not None and isinstance(feat, torch.Tensor):
                feat = feat.reshape(H, W, -1)  # [H, W, D]
                # Downsample
                feat = feat.permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
                feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                feat = feat.squeeze(0).permute(1, 2, 0)  # [H', W', D]
            else:
                feat = torch.randn(target_h, target_w, self.output_dim, device=self.device)
            
            all_features.append(feat)
        
        return torch.stack(all_features, dim=0)  # [B, H', W', D]

    
    def forward(
        self,
        video: torch.Tensor,  # [B, C, 2, H, W]
    ) -> torch.Tensor:
        """
        Encode video frames to Concerto features.
        
        Args:
            video: Two consecutive frames [B, C, 2, H, W]
            
        Returns:
            features: [B, 2, H, W, D] Concerto features for both frames
        """
        B, C, T, H, W = video.shape
        assert T == 2, "Expected exactly 2 frames"
        
        # Encode each frame separately (for dynamic scenes)
        features_t0 = self.encode_single_frame(video[:, :, 0])
        features_t1 = self.encode_single_frame(video[:, :, 1])
        
        return torch.stack([features_t0, features_t1], dim=1)


class VideoFrameExtractor:
    """
    Utility to extract frames from video files (.webm, .mp4, etc.)
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self._open_video()
    
    def _open_video(self):
        """Open video file with OpenCV."""
        try:
            import cv2
            self.cap = cv2.VideoCapture(self.video_path)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Opened video: {self.video_path}")
            print(f"  Frames: {self.frame_count}, FPS: {self.fps}, Size: {self.width}x{self.height}")
        except ImportError:
            print("Warning: OpenCV not installed. Install with: pip install opencv-python")
            self.cap = None
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the video.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            RGB numpy array [H, W, 3] or None if failed
        """
        if self.cap is None:
            return None
        
        import cv2
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            # Convert BGR to RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def get_frame_pair(
        self, 
        first_idx: int, 
        offset: int = 5,
        target_size: Tuple[int, int] = (256, 256),
    ) -> Optional[torch.Tensor]:
        """
        Get a pair of frames as a tensor.
        
        Args:
            first_idx: Index of first frame
            offset: Frame offset for second frame
            target_size: (H, W) to resize to
            
        Returns:
            Tensor [C, 2, H, W] or None if failed
        """
        import cv2
        
        frame1 = self.get_frame(first_idx)
        frame2 = self.get_frame(min(first_idx + offset, self.frame_count - 1))
        
        if frame1 is None or frame2 is None:
            return None
        
        # Resize
        frame1 = cv2.resize(frame1, target_size[::-1])  # cv2 uses (W, H)
        frame2 = cv2.resize(frame2, target_size[::-1])
        
        # Convert to tensor [C, H, W] in [0, 1] range
        frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float() / 255.0
        frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float() / 255.0
        
        # Stack as [C, 2, H, W]
        return torch.stack([frame1, frame2], dim=1)
    
    def close(self):
        if self.cap is not None:
            self.cap.release()
    
    def __del__(self):
        self.close()


# Legacy classes for backward compatibility
class DepthEstimator(nn.Module):
    """Legacy depth estimator - now handled by VGGT."""
    
    def __init__(self, model_type: str = "vggt", device: str = "cuda"):
        super().__init__()
        print("Note: DepthEstimator is deprecated. Using VGGT for depth estimation.")
        self.vggt = VGGTEncoder(device=device) if model_type == "vggt" else None
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.vggt is not None:
            results = self.vggt.encode_single_frame(images)
            return results['depth_map']
        return torch.ones(images.shape[0], images.shape[2], images.shape[3])


class PointCloudLifter(nn.Module):
    """Legacy point cloud lifter - now handled by VGGT."""
    
    def __init__(self, **kwargs):
        super().__init__()
        print("Note: PointCloudLifter is deprecated. Using VGGT for point cloud generation.")
    
    def forward(self, rgb, depth, **kwargs):
        print("Warning: Using legacy PointCloudLifter. Consider using VGGTEncoder instead.")
        B, C, H, W = rgb.shape
        
        # Simple back-projection with default intrinsics
        fx = fy = 500.0
        cx, cy = W / 2, H / 2
        
        u = torch.arange(W, device=rgb.device).float()
        v = torch.arange(H, device=rgb.device).float()
        u, v = torch.meshgrid(u, v, indexing='xy')
        
        z = depth
        x = (u.unsqueeze(0) - cx) * z / fx
        y = (v.unsqueeze(0) - cy) * z / fy
        
        coords = torch.stack([x, y, z], dim=-1)
        colors = rgb.permute(0, 2, 3, 1)
        
        return {
            "coord": coords,
            "color": colors,
        }
