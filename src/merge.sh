#!/bin/bash
set -e

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DIFFUSERS_DISABLE_BITSANDBYTES=1
export CUDA_VISIBLE_DEVICES=1

MODEL_TYPE="DiT-g/2"
IMAGE_SIZE=1024
RANK=32

MODEL_ROOT="/data/user/yyh_dk/HunyuanDiT-main/ckpts/models--Tencent-Hunyuan--HunyuanDiT-v1.2/snapshots/9f341259956a54419e407b93c3282756c14b9930"

BASE_CKPT="/data/private/nfs_ds1/yyh/HyDiT_trained_model/merge_model/SAR100K/hy1.2_merge_sar100k_rank32_ckpt12000.pt"

LORA_CKPT="/data/private/nfs_ds1/yyh/HyDiT_trained_model/001-lora_porcelain_ema_sar100k_mini_rank32/checkpoints/0016800.pt"

OUTPUT_MERGE_PATH="/data/private/nfs_ds1/yyh/HyDiT_trained_model/merge_model/SAR100K/mthy1.2_merge_sar100k_mini_1.68w.pt"

echo "[INFO] Start merging LoRA into base model"
echo "[INFO] Base model root: ${MODEL_ROOT}"
echo "[INFO] Base ckpt: ${BASE_CKPT}"
echo "[INFO] LoRA ckpt: ${LORA_CKPT}"
echo "[INFO] Output: ${OUTPUT_MERGE_PATH}"

PYTHONPATH=./ python lora/merge.py \
  --model ${MODEL_TYPE} \
  --image-size ${IMAGE_SIZE} \
  --model-root ${MODEL_ROOT} \
  --base-ckpt ${BASE_CKPT} \
  --lora-ckpt ${LORA_CKPT} \
  --rank ${RANK} \
  --multireso \
  --reso-step 64 \
  --qk-norm \
  --rope-img base1024 \
  --rope-real \
  --output-merge-path ${OUTPUT_MERGE_PATH}

echo "[INFO] Merge finished successfully"