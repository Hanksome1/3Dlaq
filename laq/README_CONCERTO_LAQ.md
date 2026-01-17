# Concerto-LAQ: 2D+3D Aware Latent Action Quantization

This project integrates **Concerto** (joint 2D-3D self-supervised learning) with **LAQ** (Latent Action Quantization) to predict actions in a latent space with 3D awareness.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Concerto-LAQ Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Video Frame (t)              Video Frame (t+1)                       │
│        │                             │                                  │
│        ▼                             ▼                                  │
│   ┌─────────┐                   ┌─────────┐                            │
│   │  VGGT   │                   │  VGGT   │  ← Single-frame depth &    │
│   │         │                   │         │    camera estimation       │
│   └────┬────┘                   └────┬────┘                            │
│        │                             │                                  │
│        ▼                             ▼                                  │
│   Point Cloud                   Point Cloud                            │
│   [8K points]                   [8K points]  ← Subsampled              │
│        │                             │                                  │
│        ▼                             ▼                                  │
│   ┌─────────┐                   ┌─────────┐                            │
│   │Concerto │                   │Concerto │  ← PTv3 3D Transformer     │
│   │ (PTv3)  │                   │ (PTv3)  │    features                │
│   └────┬────┘                   └────┬────┘                            │
│        │                             │                                  │
│        ▼                             ▼                                  │
│   Features [512]                Features [512]                         │
│        │                             │                                  │
│        └──────────┬──────────────────┘                                 │
│                   ▼                                                     │
│            ┌────────────┐                                              │
│            │  Temporal  │  ← Compute difference between                │
│            │   Encoder  │    frame features                            │
│            └─────┬──────┘                                              │
│                  │                                                      │
│                  ▼                                                      │
│            ┌────────────┐                                              │
│            │    NSVQ    │  ← Normalize-Scale VQ for                    │
│            │ Quantizer  │    discrete action codes                     │
│            └─────┬──────┘                                              │
│                  │                                                      │
│                  ▼                                                      │
│         Action Codes [4]  ← 4 discrete tokens per frame pair           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. VGGT (Visual Geometry Grounded Transformer)
- **Purpose**: Single-frame depth and camera intrinsic/extrinsic estimation
- **Input**: RGB image [224×224]
- **Output**: 
  - Depth map [224×224]
  - Point cloud [224×224, 3]
  - Camera intrinsics [3×3]
  - Camera extrinsics [4×4]

### 2. Concerto (PTv3)
- **Purpose**: Extract 2D+3D aware features from point clouds
- **Input**: Point cloud [8K points, 9 features (coord + color + normal)]
- **Output**: Per-point features [N, 512]
- **Processing**: Global average pooling → [512] feature vector

### 3. LAQ (Latent Action Quantization)
- **Purpose**: Quantize continuous features into discrete action codes
- **Components**:
  - Feature projection: 512 → model_dim
  - Spatial encoder: Transformer for spatial features
  - Temporal encoder: Compute difference between frames
  - NSVQ Quantizer: Normalize-Scale Vector Quantization
- **Output**: 4 discrete action tokens per frame pair

## Installation

### Dependencies
```bash
# Core dependencies
pip install torch torchvision opencv-python einops

# VGGT
git clone https://github.com/facebookresearch/vggt.git
cd vggt && pip install . && cd ..

# Concerto
git clone https://github.com/Pointcept/Concerto.git
cd Concerto && pip install -e . && cd ..

# Additional for Concerto
pip install spconv-cu121  # Match your CUDA version
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

### Environment Setup
```bash
export PYTHONPATH=/path/to/vggt:/path/to/Concerto:$PYTHONPATH
```

## Usage

### Training
```bash
# Quick test
python laq/train_concerto_laq.py \
    --data_dir /path/to/something-something-v2 \
    --max_videos 100 \
    --batch_size 2 \
    --num_steps 1000

# Full training
python laq/train_concerto_laq.py \
    --data_dir /path/to/something-something-v2 \
    --batch_size 8 \
    --num_steps 100000 \
    --use_wandb
```

### Visualization
```bash
# Visualize point cloud from video
python laq/visualize_vggt_pointcloud.py \
    --video /path/to/video.webm \
    --output output_viz/ \
    --frame 30
```

## Key Files

| File | Description |
|------|-------------|
| `laq/laq_model/concerto_wrapper.py` | VGGT + Concerto encoder integration |
| `laq/laq_model/concerto_laq.py` | Main ConcertoLAQ model |
| `laq/laq_model/webm_dataset.py` | Dataset for .webm video files |
| `laq/train_concerto_laq.py` | Training script |
| `laq/visualize_vggt_pointcloud.py` | Point cloud visualization |

## Configuration

### Model Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `concerto_model` | `concerto_base` | Concerto model size |
| `dim` | 512 | Model hidden dimension |
| `codebook_size` | 256 | Number of discrete action codes |
| `code_seq_len` | 4 | Action tokens per frame pair |
| `spatial_depth` | 4 | Spatial transformer layers |
| `temporal_depth` | 4 | Temporal transformer layers |

### Data Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_size` | 224 | Input frame size (must be divisible by 14) |
| `frame_offset` | 5 | Frames between pair |
| `max_points` | 8192 | Max points after subsampling |

## Pipeline Details

### Point Cloud Processing
1. **VGGT Encoding**: RGB → Depth + Camera → Point Cloud
2. **Subsampling**: 50K → 8K points (random sampling, FPS available)
3. **Normalization**: Center and scale to [-1, 1]
4. **Grid Voxelization**: For sparse convolution

### Feature Extraction
1. **Concerto Forward**: Point cloud → Per-point features [N, 512]
2. **Global Pooling**: Average pool → [512] feature vector
3. **Feature Projection**: [512] → [model_dim]

### Action Quantization
1. **Temporal Difference**: features_t1 - features_t0
2. **NSVQ Quantization**: Continuous → Discrete codes
3. **Output**: 4 action tokens representing the motion

## Known Limitations

1. **Global Pooling**: Currently loses spatial information. Future work: use sparse-to-dense projection.
2. **No Surface Normals**: VGGT doesn't provide normals; using zeros as placeholder.
3. **Memory Usage**: VGGT + Concerto requires ~40GB GPU memory for batch_size=2.

## References

- [VGGT](https://github.com/facebookresearch/vggt): Visual Geometry Grounded Transformer
- [Concerto](https://github.com/Pointcept/Concerto): Joint 2D-3D Self-Supervised Learning
- [PTv3](https://github.com/Pointcept/PointTransformerV3): Point Transformer V3
