#!/bin/bash
set -euo pipefail

# =====================================================
# GPU / Runtime
# =====================================================
export CUDA_VISIBLE_DEVICES=2
export TORCH_CUDA_ARCH_LIST="8.0"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# =====================================================
# Conda / bitsandbytes / lib path
# =====================================================
export DIFFUSERS_DISABLE_BITSANDBYTES=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# =====================================================
# HuggingFace / RamiGen Cache
# =====================================================
export HF_HOME=/data/private/nfs_ds1/yyh/pretrained_model_.cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

# =====================================================
# Project paths
# =====================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_ROOT=ckpts/models--Tencent-Hunyuan--HunyuanDiT-v1.2/snapshots/9f341259956a54419e407b93c3282756c14b9930

# Custom base DiT which fine-tuned for the specific task and merged with original weights
CUSTOM_BASE_DIT=""

# /data/private/nfs_ds1/yyh/HyDiT_trained_model/merge_model/Tank/mthy1.2_merge_Tank_rank64_3w.pt
CUSTOM_BASE_LOAD_KEY=merge

# =====================================================
# Condition-fusion ControlNet
# =====================================================
FUSION_CONTROLNET_WEIGHT=$PROJECT_ROOT/log_EXP/ControlNet/..
FUSION_CONTROL_TYPES=(edge depth mask)
FUSION_IN_CHANNELS=7
FUSION_CONTROL_WEIGHT="1.0"


# =====================================================
# Shared inference params
# =====================================================
IMAGE_SIZE="1024 1024"
INFER_STEPS=30
CFG_SCALE=6.5
BATCH_SIZE=1
SEED=42
NEGATIVE_PROMPT="blurry, low quality, distortion, lowres, bad anatomy"
MAX_GENERATE_IMAGE=12000

# =====================================================
# Dataset / output
# =====================================================

FUSION_JSONL=/data/private/nfs_ds1/yyh/dataset/Denoising_config_dataset/Controlnet/Crater_denoising_process_test_with_resolution.jsonl
# /data/private/nfs_ds1/yyh/dataset/Denoising_config_dataset/Controlnet/Tank_denoising_process_test_controlnet.jsonl
# /data/private/nfs_ds1/yyh/dataset/Denoising_config_dataset/Controlnet/Crater_denoising_process_test_controlnet.jsonl
FUSION_OUTDIR=$PROJECT_ROOT/results/...
# =====================================================
# Run
# =====================================================
cd "$PROJECT_ROOT"
export PYTHONPATH=.

CMD=(
  python3 "$PROJECT_ROOT/sample_controlnet_multi.py"

  # Basic model
  --model-root "$MODEL_ROOT"
  --dit-weight "$CUSTOM_BASE_DIT"
  --load-key "$CUSTOM_BASE_LOAD_KEY"
  --rope-img base1024 
  --rope-real 

  # Inference
  --image-size $IMAGE_SIZE
  --infer-steps "$INFER_STEPS"
  --cfg-scale "$CFG_SCALE"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --negative "$NEGATIVE_PROMPT"
  --sampler dpmms
  --max-generate-image "$MAX_GENERATE_IMAGE"
  --no-enhance
  # --skip-existing

  # Multi-resolution generation
  --auto-bucket
  --bucket-source-json "/data/user/yyh_dk/RamiGen_Gate_Dinov3/dataset/porcelain_military/jsons/porcelain_mt.json"
  --bucket-min-count 1
  --bucket-area-weight 0.05

  # Dataset / output
  --results-dir "$FUSION_OUTDIR"
  --index-file "$FUSION_JSONL"

  # One fused ControlNet
  --controlnet-weight "$FUSION_CONTROLNET_WEIGHT"
  --control-weight "$FUSION_CONTROL_WEIGHT"

  # Condition fusion
  --use-condition-fusion
  --fusion-control-types "${FUSION_CONTROL_TYPES[@]}"
  --condition-fusion-in-channels "$FUSION_IN_CHANNELS"
)

CMD+=(--infer-mode fa)

echo "================================================"
echo "Latent-space condition-fusion ControlNet inference"
echo "================================================"
echo "MODEL_ROOT                = $MODEL_ROOT"
echo "CUSTOM_BASE_DIT           = $CUSTOM_BASE_DIT"
echo "FUSION_CONTROLNET_WEIGHT  = $FUSION_CONTROLNET_WEIGHT"
echo "FUSION_JSONL              = $FUSION_JSONL"
echo "FUSION_OUTDIR             = $FUSION_OUTDIR"
echo "IMAGE_SIZE                = $IMAGE_SIZE"
echo "INFER_STEPS               = $INFER_STEPS"
echo "CFG_SCALE                 = $CFG_SCALE"
echo "BATCH_SIZE                = $BATCH_SIZE"
echo "SEED                      = $SEED"
echo "FUSION_CONTROL_TYPES      = ${FUSION_CONTROL_TYPES[*]}"
echo "FUSION_IN_CHANNELS        = $FUSION_IN_CHANNELS"
echo "FUSION_CONTROL_WEIGHT     = $FUSION_CONTROL_WEIGHT"
# multi-resolution
echo "AUTO_BUCKET               = enabled"
echo "BUCKET_SOURCE_JSON        = $PROJECT_ROOT/dataset/porcelain_military/jsons/porcelain_mt.json"
echo "BUCKET_MIN_COUNT          = 1"
echo "BUCKET_AREA_WEIGHT        = 0.05"
echo "ROPE_IMG                  = base1024"
echo "ROPE_REAL                 = enabled"
echo "================================================"

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"

echo
echo "Inference finished."
echo "Results saved to: $FUSION_OUTDIR"
