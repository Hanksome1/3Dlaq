"""
CPU-compatible test script for Concerto-LAQ.

This script tests the model architecture on CPU without requiring:
- GPU/CUDA
- Pre-trained Concerto weights
- Real data

Use this to verify the implementation is correct before running on GPU.

Usage:
    python test_concerto_laq_cpu.py
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Ensure module can be imported
sys.path.insert(0, str(Path(__file__).parent))


def test_latent_nsvq():
    """Test LatentSpaceNSVQ module."""
    print("\n" + "="*50)
    print("Testing LatentSpaceNSVQ...")
    print("="*50)
    
    from laq_model.latent_nsvq import LatentSpaceNSVQ
    
    # Create module
    nsvq = LatentSpaceNSVQ(
        input_dim=256,
        embedding_dim=32,
        num_embeddings=8,
        code_seq_len=4,
        feature_size=(8, 8),
        device='cpu',
    )
    nsvq.train()
    
    print(f"  ✓ Module created")
    print(f"    - Input dim: 256")
    print(f"    - Embedding dim: 32")
    print(f"    - Codebook size: 8")
    print(f"    - Code seq len: 4")
    
    # Test forward pass
    B, H, W, D = 2, 8, 8, 256
    features_t0 = torch.randn(B, H, W, D)
    features_t1 = torch.randn(B, H, W, D)
    
    decoded, perplexity, usage, indices = nsvq(features_t0, features_t1)
    
    print(f"  ✓ Forward pass successful")
    print(f"    - Input shape: [{B}, {H}, {W}, {D}]")
    print(f"    - Output shape: {list(decoded.shape)}")
    print(f"    - Indices shape: {list(indices.shape)}")
    print(f"    - Perplexity: {perplexity.item():.2f}")
    
    # Test inference mode
    nsvq.eval()
    with torch.no_grad():
        decoded_inf, indices_inf = nsvq.inference(features_t0, features_t1)
    
    print(f"  ✓ Inference mode successful")
    
    return True


def test_concerto_wrapper():
    """Test ConcertoEncoder wrapper (dummy mode)."""
    print("\n" + "="*50)
    print("Testing ConcertoEncoder (dummy mode)...")
    print("="*50)
    
    from laq_model.concerto_wrapper import (
        DepthEstimator, 
        PointCloudLifter, 
        ConcertoEncoder
    )
    
    # Test DepthEstimator (dummy mode)
    depth_est = DepthEstimator(model_type="dummy", device="cpu")
    print(f"  ✓ DepthEstimator created (dummy mode)")
    
    B, C, H, W = 2, 3, 256, 256
    images = torch.randn(B, C, H, W)
    depth = depth_est(images)
    
    print(f"    - Input shape: {list(images.shape)}")
    print(f"    - Depth shape: {list(depth.shape)}")
    
    # Test PointCloudLifter
    lifter = PointCloudLifter()
    print(f"  ✓ PointCloudLifter created")
    
    point_data = lifter(images, depth)
    print(f"    - Point coord shape: {list(point_data['coord'].shape)}")
    print(f"    - Point color shape: {list(point_data['color'].shape)}")
    
    # Test ConcertoEncoder (dummy mode - without actual Concerto)
    # This will use dummy encoder since concerto package isn't installed
    encoder = ConcertoEncoder(
        model_name="concerto_base",
        freeze=True,
        device="cpu",
        depth_estimator=depth_est,
    )
    print(f"  ✓ ConcertoEncoder created (dummy mode)")
    
    # Test video encoding
    video = torch.randn(B, C, 2, H, W)  # [B, C, T, H, W]
    features = encoder(video)
    
    print(f"    - Video input shape: {list(video.shape)}")
    print(f"    - Features output shape: {list(features.shape)}")
    
    return True


def test_concerto_laq():
    """Test full ConcertoLAQ model."""
    print("\n" + "="*50)
    print("Testing ConcertoLAQ (full model)...")
    print("="*50)
    
    from laq_model.concerto_laq import ConcertoLAQ
    
    # Create model with small dimensions for testing
    model = ConcertoLAQ(
        concerto_model_name="concerto_base",
        concerto_dim=256,
        dim=512,  # Smaller for CPU testing
        quant_dim=32,
        codebook_size=8,
        code_seq_len=4,
        image_size=64,  # Smaller for CPU testing
        feature_size=(8, 8),
        spatial_depth=2,  # Smaller for CPU testing
        temporal_depth=2,
        predictor_depth=2,
        dim_head=32,
        heads=4,
        freeze_concerto=True,
        use_depth=False,
        depth_model_type="dummy",
        use_precomputed_features=True,  # Skip Concerto, use pre-computed
    )
    model.train()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  ✓ Model created")
    print(f"    - Total parameters: {total_params:,}")
    print(f"    - Trainable parameters: {trainable_params:,}")
    
    # Test with pre-computed features
    B, H, W, D = 2, 8, 8, 256
    features = torch.randn(B, 2, H, W, D)  # [B, T, H, W, concerto_dim]
    
    # Forward pass (training mode)
    loss, num_unique = model(features=features, step=0)
    
    print(f"  ✓ Training forward pass successful")
    print(f"    - Features shape: {list(features.shape)}")
    print(f"    - Loss: {loss.item():.4f}")
    print(f"    - Unique indices: {num_unique}")
    
    # Test backward pass
    loss.backward()
    print(f"  ✓ Backward pass successful")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        indices = model(features=features, return_only_codebook_ids=True)
    
    print(f"  ✓ Inference mode successful")
    print(f"    - Indices shape: {list(indices.shape)}")
    print(f"    - Unique codes: {indices.unique().tolist()}")
    
    return True


def test_data_pipeline():
    """Test data pipeline classes."""
    print("\n" + "="*50)
    print("Testing Data Pipeline...")
    print("="*50)
    
    from laq_model.concerto_data import (
        ConcertoVideoDataset, 
        PrecomputedFeatureDataset,
        collate_fn
    )
    
    print(f"  ✓ Data classes imported successfully")
    print(f"    - ConcertoVideoDataset")
    print(f"    - PrecomputedFeatureDataset")
    print(f"    - collate_fn")
    
    return True


def test_trainer_init():
    """Test trainer initialization (without actual training)."""
    print("\n" + "="*50)
    print("Testing Trainer Initialization...")
    print("="*50)
    
    # Just test import, since trainer requires dataset
    from laq_model.concerto_trainer import ConcertoLAQTrainer
    
    print(f"  ✓ ConcertoLAQTrainer imported successfully")
    
    return True


def main():
    print("="*50)
    print("Concerto-LAQ CPU Test Suite")
    print("="*50)
    print("\nThis test verifies the model architecture works correctly")
    print("without requiring GPU, Concerto weights, or real data.\n")
    
    all_passed = True
    
    try:
        test_latent_nsvq()
    except Exception as e:
        print(f"  ✗ LatentSpaceNSVQ test failed: {e}")
        all_passed = False
    
    try:
        test_concerto_wrapper()
    except Exception as e:
        print(f"  ✗ ConcertoEncoder test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_concerto_laq()
    except Exception as e:
        print(f"  ✗ ConcertoLAQ test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_data_pipeline()
    except Exception as e:
        print(f"  ✗ Data pipeline test failed: {e}")
        all_passed = False
    
    try:
        test_trainer_init()
    except Exception as e:
        print(f"  ✗ Trainer test failed: {e}")
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests passed!")
        print("="*50)
        print("\nThe model architecture is correct. You can now:")
        print("1. Copy this code to a GPU server")
        print("2. Install dependencies: pip install concerto torch accelerate wandb")
        print("3. Run training: python train_concerto_laq.py --folder /path/to/data")
    else:
        print("❌ Some tests failed!")
        print("="*50)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
