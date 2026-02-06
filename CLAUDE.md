# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LAPA (Latent Action Pretraining from Videos)** — an unsupervised approach for pretraining Vision-Language-Action (VLA) models without ground-truth robot action labels. The project trains a 7B LLaMA-based VLA model using latent actions derived from video frame pairs.

Paper: https://arxiv.org/abs/2410.11758

## Two Separate Frameworks

This repo contains two distinct codebases with **different ML frameworks and Python environments**:

### 1. `latent_pretraining/` — JAX/Flax (conda env: `lapa`)
The main VLA model. A 7B LLaMA variant that takes images + text instructions and predicts latent/real actions. Built on the [Large World Model (LWM)](https://github.com/LargeWorldModel/LWM) codebase using the `tux` library for distributed JAX training.

- **Install**: `pip install -r requirements.txt` (root-level)
- **Key deps**: JAX (CUDA 12), Flax 0.7.0, TensorFlow, `tux` (git dep)
- **Models**: `llama.py` (base LLaMA), `vision_llama.py` (vision variant), `delta_llama.py` / `delta_llama_action.py` (latent action variants)
- **Data**: `data.py` — dataset factory with multiple processors (vision+text, delta, action)
- **Training**: `python -m latent_pretraining.train` via shell scripts in `scripts/`
- **Inference**: `python -m latent_pretraining.inference`
- **Deployment**: `python -m latent_pretraining.deploy`

### 2. `laq/` — PyTorch (conda env: `laq`, install via `cd laq && pip install -e .`)
Latent Action Quantization module. Trains an inverse dynamics model on video frame pairs to produce discrete latent action tokens. These tokens become training targets for the main VLA model.

- **Install**: `cd laq && pip install -e .`
- **Key deps**: PyTorch 2.2.0, accelerate, vector-quantize-pytorch, timm
- **Core model code**: `laq/laq_model/` package

## Commands

### LAQ Training (PyTorch)
```bash
# Original LAQ on Something-Something v2
cd laq && accelerate launch train_sthv2.py

# Concerto-LAQ (2D+3D aware) — single GPU
python laq/train_concerto_laq.py --data_dir /path/to/videos --batch_size 2 --num_steps 100000

# Concerto-LAQ — multi-GPU with DDP
torchrun --nproc_per_node=4 laq/train_concerto_laq_ddp.py --data_dir /path/to/data --batch_size 4

# LAQ inference (generate latent actions for pretraining data)
python laq/inference_sthv2.py
```

### Latent Pretraining (JAX)
```bash
# Set absolute_path in scripts first, then:
./scripts/latent_pretrain_openx.sh    # Pretrain on Open-X (8 GPUs, --mesh_dim 2nd value = GPU count)
./scripts/finetune_real.sh            # Fine-tune on real robot data (4 GPUs)
./scripts/finetune_simpler.sh         # Fine-tune on SIMPLER sim data (4 GPUs)
```

### Data Preprocessing
```bash
python data/finetune_preprocess.py --input_path /path/to/json --output_filename data/real_finetune.jsonl --csv_filename data/real_finetune.csv
```

## Architecture Details

### LAQ Models (`laq/laq_model/`)
- `latent_action_quantization.py` — Original LAQ: ViT spatial encoder + temporal encoder + VQ (codebook_size=8, 4 code tokens per frame pair)
- `latent_nsvq.py` — NSVQ variant using Normalize-Scale Vector Quantization
- `concerto_laq.py` — Concerto-LAQ: VGGT (1B, frozen) → point clouds → Concerto/PTv3 (108M, frozen) → sparse-to-dense projection → spatial/temporal encoders → NSVQ quantizer
- `sparse_to_dense.py` — Cross-attention module converting sparse point cloud features [N, 512] to dense 2D grid [14x14, 512]
- `pq_vae.py`, `ed_vae.py` — Alternative quantization approaches (Product Quantization VAE, encoder-decoder VAE)
- `laq_trainer.py` — Original LAQ trainer; `concerto_trainer.py` — Concerto-LAQ trainer

### Latent Pretraining Models (`latent_pretraining/`)
- `llama.py` — Base LLaMA config and Flax module
- `delta_llama_action.py` — Full model with both latent action (delta) and real action heads
- `vqgan.py` — VQGAN for image tokenization (256 discrete tokens per image)
- `data.py` — Dataset factory; processors handle vision tokens, delta tokens (latent actions), and action tokens
- `train.py` — Main training loop using `tux` for JAX distributed training with model parallelism

### Key Parameters
- LAQ default: `dim=1024, codebook_size=8, patch_size=32, spatial_depth=8, temporal_depth=8, heads=16, code_seq_len=4`
- Pretraining: `--mesh_dim='!-1,8,1,1'` where 2nd value = number of GPUs; `--modality` controls which heads are active (`vision,text,delta` for pretrain, `vision,action,delta` for finetune)
- All training scripts require setting `absolute_path` variable to the project root

## External Dependencies (for Concerto-LAQ)

VGGT and Concerto are cloned into the repo root and installed in editable mode. Pretrained weights auto-download from HuggingFace on first use.

```bash
# Already cloned at /root/3Dlaq/vggt and /root/3Dlaq/Concerto
pip install -e vggt/
pip install -e Concerto/
pip install spconv-cu120 flash-attn --no-build-isolation
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install addict natsort camtools
```

- **VGGT** (`facebook/VGGT-1B`): 1B-param single-frame 3D estimation. Auto-downloads from HuggingFace.
- **Concerto** (`Pointcept/Concerto`, `concerto_base`): 108M-param PTv3 3D backbone. Auto-downloads from HuggingFace. Requires flash-attn.
- **SimplerEnv**: Simulation evaluation environment, included as subdirectory with its own setup.
- **LWM checkpoints**: Base model weights from [LWM-Chat-1M-Jax](https://huggingface.co/LargeWorldModel/LWM-Chat-1M-Jax).

## Data Format
- Training data is JSONL with fields: `id`, `image`, `instruction`, `vision` (256 VQGAN tokens), `delta` (4 latent action tokens)
- Fine-tuning data JSON contains `conversations` with `raw_actions` (7-DOF end-effector) and `states` (eef_pos, eef_euler, gripper_state)
- Videos for LAQ training: directory of `.webm` files (e.g., Something-Something v2 structure)
