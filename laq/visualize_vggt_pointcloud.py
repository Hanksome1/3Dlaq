"""
Visualization script for VGGT and Concerto pipeline.

This script:
1. Loads real VGGT model
2. Processes video frames
3. Generates 3D point clouds with depth and camera estimation
4. Saves point clouds as PLY files for visualization
5. Optionally runs through Concerto

Usage:
    python visualize_vggt_pointcloud.py --video ../dataset/159833.webm --output output/

    # View in MeshLab, Open3D, or online: https://3dviewer.net/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


def save_pointcloud_ply(points: np.ndarray, colors: np.ndarray, filename: str):
    """
    Save point cloud as PLY file.
    
    Args:
        points: [N, 3] xyz coordinates
        colors: [N, 3] RGB values (0-255)
        filename: Output PLY file path
    """
    N = points.shape[0]
    
    # Ensure colors are uint8
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    
    # Write PLY file
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(N):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"Saved point cloud to: {filename} ({N} points)")


def save_depth_image(depth: np.ndarray, filename: str):
    """Save depth map as image."""
    from PIL import Image
    
    # Normalize to 0-255
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Apply colormap (viridis-like)
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        cmap[i] = [int(255 * (0.267 + 0.733 * t)), 
                   int(255 * (0.004 + 0.873 * t - 0.477 * t**2)),
                   int(255 * (0.329 + 0.671 * t - 0.5 * t**2))]
    
    colored = cmap[depth_uint8]
    img = Image.fromarray(colored)
    img.save(filename)
    print(f"Saved depth map to: {filename}")


def test_vggt_real(video_path: str, output_dir: str, frame_idx: int = 0):
    """Test VGGT with real model."""
    print("\n" + "="*60)
    print("Testing VGGT with Real Model")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load video frame
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video frame")
        return None
    
    # Convert to RGB and resize (must be divisible by VGGT patch size 14)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (224, 224))  # 224 = 14 * 16
    
    # Save original frame
    from PIL import Image
    Image.fromarray(frame_rgb).save(os.path.join(output_dir, "input_frame.png"))
    print(f"Saved input frame to: {output_dir}/input_frame.png")
    
    # Convert to tensor
    image = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    image = image.unsqueeze(0)  # [1, 3, H, W]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load VGGT
    print("\nLoading VGGT model...")
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        
        # Determine dtype
        if device == "cuda":
            capability = torch.cuda.get_device_capability()[0]
            dtype = torch.bfloat16 if capability >= 8 else torch.float16
        else:
            dtype = torch.float32
        
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        model.eval()
        print("✓ VGGT model loaded successfully!")
        
    except Exception as e:
        print(f"✗ Failed to load VGGT: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Run VGGT
    print("\nRunning VGGT inference...")
    image = image.to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # VGGT expects [batch, num_images, C, H, W]
            images = image.unsqueeze(0)  # [1, 1, C, H, W]
            
            # Get aggregated tokens
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            
            # Predict cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            
            # Predict depth
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            
            # Predict point map
            point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
    
    print("✓ VGGT inference completed!")
    
    # Extract results (remove batch dimensions)
    depth = depth_map[0, 0].cpu().numpy()  # [H, W]
    points = point_map[0, 0].cpu().numpy()  # [H, W, 3]
    confidence = point_conf[0, 0].cpu().numpy()  # [H, W]
    intrinsic_mat = intrinsic[0, 0].cpu().numpy()  # [3, 3]
    extrinsic_mat = extrinsic[0, 0].cpu().numpy()  # [4, 4]
    
    H, W = depth.shape
    
    print(f"\nResults:")
    print(f"  Depth map: {depth.shape}, range [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"  Point map: {points.shape}")
    print(f"  Confidence: {confidence.shape}, range [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"\n  Camera intrinsics:")
    print(f"    fx={intrinsic_mat[0,0]:.2f}, fy={intrinsic_mat[1,1]:.2f}")
    print(f"    cx={intrinsic_mat[0,2]:.2f}, cy={intrinsic_mat[1,2]:.2f}")
    
    # Save depth map
    save_depth_image(depth, os.path.join(output_dir, "depth_map.png"))
    
    # Save confidence map
    save_depth_image(confidence, os.path.join(output_dir, "confidence_map.png"))
    
    # Prepare point cloud
    points_flat = points.reshape(-1, 3)  # [N, 3]
    colors_flat = frame_rgb.reshape(-1, 3)  # [N, 3]
    confidence_flat = confidence.reshape(-1)
    
    # Filter by confidence (optional, remove low-confidence points)
    conf_threshold = 0.1
    valid_mask = confidence_flat > conf_threshold
    points_filtered = points_flat[valid_mask]
    colors_filtered = colors_flat[valid_mask]
    
    print(f"\n  Total points: {len(points_flat)}")
    print(f"  Valid points (conf > {conf_threshold}): {len(points_filtered)}")
    
    # Save full point cloud
    save_pointcloud_ply(
        points_flat, colors_flat,
        os.path.join(output_dir, "pointcloud_full.ply")
    )
    
    # Save filtered point cloud
    save_pointcloud_ply(
        points_filtered, colors_filtered,
        os.path.join(output_dir, "pointcloud_filtered.ply")
    )
    
    # Save camera parameters
    np.savez(
        os.path.join(output_dir, "camera_params.npz"),
        intrinsic=intrinsic_mat,
        extrinsic=extrinsic_mat,
        depth=depth,
        confidence=confidence,
    )
    print(f"Saved camera parameters to: {output_dir}/camera_params.npz")
    
    return {
        'depth': depth,
        'points': points,
        'confidence': confidence,
        'intrinsic': intrinsic_mat,
        'extrinsic': extrinsic_mat,
        'colors': frame_rgb,
    }


def test_concerto_real(vggt_results: dict, output_dir: str):
    """Test Concerto with real model."""
    print("\n" + "="*60)
    print("Testing Concerto with Real Model")
    print("="*60)
    
    try:
        import pointcept.models.point_transformer_v3.point_transformer_v3 as ptv3
        print("✓ Pointcept/PTv3 available")
    except ImportError:
        print("✗ Pointcept not installed")
        print("  Install with: pip install pointcept")
        return None
    
    try:
        import concerto
        encoder = concerto.model.load("concerto_base", repo_id="Pointcept/Concerto")
        print("✓ Concerto model loaded!")
    except Exception as e:
        print(f"✗ Failed to load Concerto: {e}")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    encoder.eval()
    
    # Prepare point cloud input
    points = vggt_results['points'].reshape(-1, 3)  # [N, 3]
    colors = vggt_results['colors'].reshape(-1, 3) / 255.0  # [N, 3]
    
    point_dict = {
        "coord": torch.from_numpy(points).float().to(device),
        "color": torch.from_numpy(colors).float().to(device),
    }
    
    print("\nRunning Concerto inference...")
    with torch.no_grad():
        output = encoder(point_dict)
    
    if isinstance(output, dict):
        features = output.get("feat", output.get("features"))
    else:
        features = output
    
    print(f"✓ Concerto inference completed!")
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature dim: {features.shape[-1]}")
    
    # Save features
    torch.save(features.cpu(), os.path.join(output_dir, "concerto_features.pt"))
    print(f"Saved features to: {output_dir}/concerto_features.pt")
    
    return features


def main():
    parser = argparse.ArgumentParser(description='Visualize VGGT point clouds')
    parser.add_argument('--video', type=str, default='../dataset/159833.webm',
                        help='Path to video file')
    parser.add_argument('--output', type=str, default='output_visualization',
                        help='Output directory for visualizations')
    parser.add_argument('--frame', type=int, default=30,
                        help='Frame index to process')
    parser.add_argument('--skip_concerto', action='store_true',
                        help='Skip Concerto encoding')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VGGT + Concerto Visualization")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Frame: {args.frame}")
    
    # Test VGGT
    vggt_results = test_vggt_real(args.video, args.output, args.frame)
    
    if vggt_results is None:
        print("\n❌ VGGT test failed")
        return 1
    
    # Test Concerto
    if not args.skip_concerto:
        concerto_features = test_concerto_real(vggt_results, args.output)
    
    print("\n" + "="*60)
    print("✅ Visualization Complete!")
    print("="*60)
    print(f"\nOutput files in: {args.output}/")
    print("  - input_frame.png       : Original RGB frame")
    print("  - depth_map.png         : Estimated depth (colored)")
    print("  - confidence_map.png    : Point confidence")
    print("  - pointcloud_full.ply   : Full point cloud")
    print("  - pointcloud_filtered.ply : Filtered by confidence")
    print("  - camera_params.npz     : Camera intrinsics/extrinsics")
    if not args.skip_concerto:
        print("  - concerto_features.pt  : Concerto features")
    
    print("\nTo visualize the point cloud:")
    print("  1. Open https://3dviewer.net/ in browser")
    print("  2. Drag and drop pointcloud_filtered.ply")
    print("  3. Or use MeshLab/Open3D for local viewing")
    
    return 0


if __name__ == "__main__":
    exit(main())
