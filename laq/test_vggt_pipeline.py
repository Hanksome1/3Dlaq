"""
Test script for Concerto-LAQ pipeline with VGGT.

This script tests the complete pipeline:
1. Load video (.webm) file
2. Extract frame pairs
3. Run VGGT for depth + camera estimation
4. Generate point clouds
5. (Optional) Run Concerto for features

Usage:
    # On GPU server with all dependencies:
    python test_vggt_pipeline.py --video /path/to/video.webm

    # Test with dummy mode (no GPU/models):
    python test_vggt_pipeline.py --video /path/to/video.webm --dummy
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_video_loading(video_path: str):
    """Test video frame extraction."""
    print("\n" + "="*60)
    print("Step 1: Testing Video Loading")
    print("="*60)
    
    try:
        import cv2
        print(f"✓ OpenCV available")
    except ImportError:
        print("✗ OpenCV not installed. Install with: pip install opencv-python")
        return None
    
    from laq_model.concerto_wrapper import VideoFrameExtractor
    
    extractor = VideoFrameExtractor(video_path)
    
    if extractor.cap is None:
        print("✗ Failed to open video")
        return None
    
    print(f"✓ Video opened successfully")
    print(f"  - Total frames: {extractor.frame_count}")
    print(f"  - FPS: {extractor.fps}")
    print(f"  - Resolution: {extractor.width}x{extractor.height}")
    
    # Get a frame pair
    frame_pair = extractor.get_frame_pair(first_idx=0, offset=5, target_size=(256, 256))
    
    if frame_pair is not None:
        print(f"✓ Extracted frame pair: {list(frame_pair.shape)}")
        return frame_pair
    else:
        print("✗ Failed to extract frames")
        return None


def test_vggt_encoding(frame_pair, use_dummy: bool = False):
    """Test VGGT encoding."""
    print("\n" + "="*60)
    print("Step 2: Testing VGGT Encoding")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("✗ PyTorch not installed")
        return None
    
    from laq_model.concerto_wrapper import VGGTEncoder
    
    device = "cuda" if torch.cuda.is_available() and not use_dummy else "cpu"
    
    # Create encoder
    if use_dummy:
        print("Note: Running in dummy mode (no actual VGGT)")
        encoder = VGGTEncoder(device="cpu")
        encoder.model = None  # Force dummy mode
    else:
        encoder = VGGTEncoder(device=device)
    
    # Move frame to device
    frame_t0 = frame_pair[:, 0].unsqueeze(0).to(device)  # [1, C, H, W]
    
    print(f"Input frame shape: {list(frame_t0.shape)}")
    
    # Encode
    results = encoder.encode_single_frame(frame_t0)
    
    print(f"✓ VGGT encoding successful!")
    print(f"  - depth_map: {list(results['depth_map'].shape)}")
    print(f"  - intrinsic: {list(results['intrinsic'].shape)}")
    print(f"  - extrinsic: {list(results['extrinsic'].shape)}")
    print(f"  - point_map: {list(results['point_map'].shape)}")
    
    if not use_dummy:
        # Print some actual values
        print(f"\n  Camera intrinsics (estimated):")
        intri = results['intrinsic'][0]
        print(f"    fx={intri[0,0].item():.2f}, fy={intri[1,1].item():.2f}")
        print(f"    cx={intri[0,2].item():.2f}, cy={intri[1,2].item():.2f}")
        
        depth = results['depth_map'][0]
        print(f"\n  Depth statistics:")
        print(f"    min={depth.min().item():.3f}, max={depth.max().item():.3f}, mean={depth.mean().item():.3f}")
    
    return results


def test_concerto_encoding(frame_pair, vggt_results, use_dummy: bool = False):
    """Test Concerto encoding."""
    print("\n" + "="*60)
    print("Step 3: Testing Concerto Encoding")
    print("="*60)
    
    try:
        import torch
    except ImportError:
        print("✗ PyTorch not installed")
        return None
    
    from laq_model.concerto_wrapper import ConcertoEncoder
    
    device = "cuda" if torch.cuda.is_available() and not use_dummy else "cpu"
    
    # Create encoder
    encoder = ConcertoEncoder(
        concerto_model_name="concerto_base",
        device=device,
    )
    
    # Move frames
    video = frame_pair.unsqueeze(0).to(device)  # [1, C, 2, H, W]
    
    print(f"Input video shape: {list(video.shape)}")
    
    # Encode
    features = encoder(video)
    
    print(f"✓ Concerto encoding successful!")
    print(f"  Output features: {list(features.shape)}")
    print(f"  Feature dim: {features.shape[-1]}")
    
    return features


def test_full_pipeline(frame_pair, use_dummy: bool = False):
    """Test the full ConcertoLAQ pipeline."""
    print("\n" + "="*60)
    print("Step 4: Testing Full ConcertoLAQ Pipeline")
    print("="*60)
    
    try:
        import torch
    except ImportError:
        print("✗ PyTorch not installed")
        return
    
    from laq_model.concerto_laq import ConcertoLAQ
    
    device = "cuda" if torch.cuda.is_available() and not use_dummy else "cpu"
    
    # Create model with small config for testing
    model = ConcertoLAQ(
        concerto_model_name="concerto_base",
        concerto_dim=256,
        dim=512,  # Smaller for testing
        quant_dim=32,
        codebook_size=8,
        code_seq_len=4,
        image_size=256,
        feature_size=(8, 8),
        spatial_depth=2,
        temporal_depth=2,
        predictor_depth=2,
        dim_head=32,
        heads=4,
        use_precomputed_features=False,  # Use VGGT directly
    )
    
    if device == "cuda":
        model = model.cuda()
    
    model.train()
    
    # Move frames
    video = frame_pair.unsqueeze(0).to(device)  # [1, C, 2, H, W]
    
    print(f"Input video shape: {list(video.shape)}")
    
    # Forward pass
    loss, num_unique = model(video=video, step=0)
    
    print(f"✓ Full pipeline successful!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Unique codes: {num_unique}")
    
    # Test inference
    model.eval()
    with torch.no_grad():
        indices = model(video=video, return_only_codebook_ids=True)
    
    print(f"  Action indices: {indices.tolist()}")


def main():
    parser = argparse.ArgumentParser(description='Test VGGT + Concerto pipeline')
    parser.add_argument('--video', type=str, default='/home/hanksome/workspace/LAPA/dataset/159833.webm',
                        help='Path to video file')
    parser.add_argument('--dummy', action='store_true',
                        help='Run in dummy mode (no GPU/models required)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Concerto-LAQ Pipeline Test")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Dummy mode: {args.dummy}")
    
    # Step 1: Load video
    frame_pair = test_video_loading(args.video)
    if frame_pair is None:
        print("\n❌ Video loading failed. Exiting.")
        return 1
    
    # Step 2: Test VGGT
    try:
        vggt_results = test_vggt_encoding(frame_pair, use_dummy=args.dummy)
    except Exception as e:
        print(f"\n❌ VGGT encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Test Concerto
    try:
        features = test_concerto_encoding(frame_pair, vggt_results, use_dummy=args.dummy)
    except Exception as e:
        print(f"\n❌ Concerto encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: Test full pipeline
    try:
        test_full_pipeline(frame_pair, use_dummy=args.dummy)
    except Exception as e:
        print(f"\n❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
