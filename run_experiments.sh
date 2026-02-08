#!/bin/bash
# Run 3 Concerto-LAQ experiments sequentially on 4x L40 GPUs
# v2: Fixed training loop + larger codebook (64) + larger batch (32)
# Logs and checkpoints saved to /mnt/nfs/hanksome/3dlaq/

set -e

export CUDA_HOME=/root/miniconda3/envs/laq
export PATH=/root/miniconda3/envs/laq/bin:/usr/bin:/bin:$PATH
DATA_DIR="/mnt/nfs/eson/dataset/20bn-something-something-v2"
BASE_DIR="/mnt/nfs/hanksome/3dlaq"
LOG_DIR="${BASE_DIR}/logs_v2"
CKPT_DIR="${BASE_DIR}/checkpoints_v2"

mkdir -p "${CKPT_DIR}/pq_vae" "${CKPT_DIR}/ed_vae" "${CKPT_DIR}/gumbel_baseline" "${LOG_DIR}"

cd /root/3Dlaq

# Pull latest fixes (training loop + codebook reset)
git pull origin main || echo "Git pull failed, using local version"

echo "=============================================="
echo "Experiment A: PQ-VAE (started $(date))"
echo "=============================================="
torchrun --nproc_per_node=4 laq/train_concerto_laq_ddp.py \
    --data_dir "$DATA_DIR" \
    --batch_size 32 \
    --codebook_size 64 \
    --lr 1e-4 \
    --vq_type pq \
    --num_steps 100005 \
    --output_dir "${CKPT_DIR}/pq_vae" \
    --log_dir "${LOG_DIR}" \
    --use_tensorboard \
    --wandb_run "concerto_laq_pq_vae_v2" \
    --save_every 5000 \
    --log_every 100 \
    2>&1 | tee "${LOG_DIR}/exp_a_pq_vae.log"

echo "=============================================="
echo "Experiment B: edVAE (started $(date))"
echo "=============================================="
torchrun --nproc_per_node=4 laq/train_concerto_laq_ddp.py \
    --data_dir "$DATA_DIR" \
    --batch_size 32 \
    --codebook_size 64 \
    --lr 1e-4 \
    --vq_type ed \
    --num_steps 100005 \
    --output_dir "${CKPT_DIR}/ed_vae" \
    --log_dir "${LOG_DIR}" \
    --use_tensorboard \
    --wandb_run "concerto_laq_ed_vae_v2" \
    --save_every 5000 \
    --log_every 100 \
    2>&1 | tee "${LOG_DIR}/exp_b_ed_vae.log"

echo "=============================================="
echo "Experiment C: Baseline Gumbel-VQ (started $(date))"
echo "=============================================="
torchrun --nproc_per_node=4 laq/train_concerto_laq_ddp.py \
    --data_dir "$DATA_DIR" \
    --batch_size 32 \
    --codebook_size 64 \
    --lr 1e-4 \
    --vq_type gumbel \
    --num_steps 100005 \
    --output_dir "${CKPT_DIR}/gumbel_baseline" \
    --log_dir "${LOG_DIR}" \
    --use_tensorboard \
    --wandb_run "concerto_laq_baseline_v2" \
    --save_every 5000 \
    --log_every 100 \
    2>&1 | tee "${LOG_DIR}/exp_c_gumbel_baseline.log"

echo "=============================================="
echo "All 3 experiments complete! ($(date))"
echo "=============================================="
echo "Checkpoints: ${CKPT_DIR}/"
echo "TensorBoard logs: ${LOG_DIR}/"
echo "Run: tensorboard --logdir ${LOG_DIR} --bind_all"

