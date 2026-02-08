#!/bin/bash
# Run Concerto-LAQ experiments with precomputed features on 4x L40 GPUs
# v3: Precompute VGGT+Concerto features once, then train LAQ head only
# ~3-4 hours precompute + ~hours per experiment (vs ~days without precompute)

set -e

export CUDA_HOME=/root/miniconda3/envs/laq
export PATH=/root/miniconda3/envs/laq/bin:/usr/bin:/bin:$PATH
DATA_DIR="/mnt/nfs/eson/dataset/20bn-something-something-v2"
BASE_DIR="/mnt/nfs/hanksome/3dlaq"
FEATURES_DIR="/mnt/nfs/hanksome/precomputed_features"
LOG_DIR="${BASE_DIR}/logs_v3"
CKPT_DIR="${BASE_DIR}/checkpoints_v3"
DONE_SENTINEL="${FEATURES_DIR}/.done"

mkdir -p "${FEATURES_DIR}" "${LOG_DIR}"
mkdir -p "${CKPT_DIR}/pq_vae" "${CKPT_DIR}/ed_vae" "${CKPT_DIR}/gumbel_baseline"

cd /root/3Dlaq

# ===== Step 0: Precompute features (skip if already done) =====
if [ ! -f "${DONE_SENTINEL}" ]; then
    echo "=============================================="
    echo "Step 0: Precomputing VGGT+Concerto features ($(date))"
    echo "=============================================="
    torchrun --nproc_per_node=4 laq/precompute_features.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$FEATURES_DIR" \
        --batch_size 4 \
        --num_workers 4 \
        2>&1 | tee "${LOG_DIR}/precompute.log"

    touch "${DONE_SENTINEL}"
    echo "Precompute complete. Sentinel: ${DONE_SENTINEL}"
else
    echo "Precomputed features found at ${FEATURES_DIR}, skipping."
fi

NUM_FEATURES=$(ls -1 "${FEATURES_DIR}"/*.pt 2>/dev/null | wc -l)
echo "Number of precomputed feature files: ${NUM_FEATURES}"

# ===== Experiment A: PQ-VAE =====
echo "=============================================="
echo "Experiment A: PQ-VAE with precomputed features ($(date))"
echo "=============================================="
torchrun --nproc_per_node=4 laq/train_concerto_laq_ddp.py \
    --use_precomputed_features \
    --features_dir "$FEATURES_DIR" \
    --batch_size 128 \
    --codebook_size 64 \
    --lr 1e-4 \
    --vq_type pq \
    --num_steps 100005 \
    --output_dir "${CKPT_DIR}/pq_vae" \
    --log_dir "${LOG_DIR}" \
    --use_tensorboard \
    --wandb_run "concerto_laq_pq_vae_v3" \
    --save_every 5000 \
    --log_every 100 \
    2>&1 | tee "${LOG_DIR}/exp_a_pq_vae.log"

# ===== Experiment B: edVAE =====
echo "=============================================="
echo "Experiment B: edVAE with precomputed features ($(date))"
echo "=============================================="
torchrun --nproc_per_node=4 laq/train_concerto_laq_ddp.py \
    --use_precomputed_features \
    --features_dir "$FEATURES_DIR" \
    --batch_size 128 \
    --codebook_size 64 \
    --lr 1e-4 \
    --vq_type ed \
    --num_steps 100005 \
    --output_dir "${CKPT_DIR}/ed_vae" \
    --log_dir "${LOG_DIR}" \
    --use_tensorboard \
    --wandb_run "concerto_laq_ed_vae_v3" \
    --save_every 5000 \
    --log_every 100 \
    2>&1 | tee "${LOG_DIR}/exp_b_ed_vae.log"

# ===== Experiment C: Baseline Gumbel-VQ =====
echo "=============================================="
echo "Experiment C: Gumbel-VQ with precomputed features ($(date))"
echo "=============================================="
torchrun --nproc_per_node=4 laq/train_concerto_laq_ddp.py \
    --use_precomputed_features \
    --features_dir "$FEATURES_DIR" \
    --batch_size 128 \
    --codebook_size 64 \
    --lr 1e-4 \
    --vq_type gumbel \
    --num_steps 100005 \
    --output_dir "${CKPT_DIR}/gumbel_baseline" \
    --log_dir "${LOG_DIR}" \
    --use_tensorboard \
    --wandb_run "concerto_laq_baseline_v3" \
    --save_every 5000 \
    --log_every 100 \
    2>&1 | tee "${LOG_DIR}/exp_c_gumbel_baseline.log"

echo "=============================================="
echo "All v3 experiments complete! ($(date))"
echo "=============================================="
echo "Features: ${FEATURES_DIR}/"
echo "Checkpoints: ${CKPT_DIR}/"
echo "TensorBoard logs: ${LOG_DIR}/"
echo "Run: tensorboard --logdir ${LOG_DIR} --bind_all"
