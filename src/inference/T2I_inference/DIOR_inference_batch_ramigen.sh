#!/bin/bash
set -euo pipefail

# =====================================================
# 🖥️ GPU / Runtime
# =====================================================
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.0"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# =====================================================
# 🧠 Conda / C++ / bitsandbytes 
# =====================================================
export DIFFUSERS_DISABLE_BITSANDBYTES=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# =====================================================
# 📦 HuggingFace / RamiGen Cache
# =====================================================
export HF_HOME=/data/private/nfs_ds1/yyh/pretrained_model_.cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

echo "📁 Cache directory : $HF_HOME"
echo "🧠 Conda prefix   : ${CONDA_PREFIX:-N/A}"
echo "🖥️ CUDA devices   : $CUDA_VISIBLE_DEVICES"

# =====================================================
# 📌 Project paths
# =====================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# RamiGen original project root（含 t2i/clip_text_encoder/mt5/...）
MODEL_ROOT=$PROJECT_ROOT/ckpts/models--Tencent-Hunyuan--HunyuanDiT-v1.2/snapshots/9f341259956a54419e407b93c3282756c14b9930

# merged DiT weights
MERGED_DIT=$PROJECT_ROOT/ckpts/HyDiT_trained_model/merge_model/DIOR/hy1.2_merge_DIOR_rank48_ckpt5000.pt

# jsonl prompt
JSONL=$PROJECT_ROOT/dataset/porcelain_DIOR/inference_jsonl/test.jsonl

# ouput directory
OUTDIR=$PROJECT_ROOT/results/RamiGen_inference_DIOR

# =====================================================
# ⚙️ inference parameters
# =====================================================
IMAGE_SIZE="1024 1024"
INFER_STEPS=30
CFG_SCALE=7.5
BATCH_SIZE=2
SEED=42

NEGATIVE_PROMPT="blurry, low quality, distortion, lowres, bad anatomy"

# =====================================================
# 🚀 Run
# =====================================================
cd "$PROJECT_ROOT"

echo "================================================"
echo "🚀 Native RamiGen jsonl batch inference"
echo "================================================"
echo "MODEL_ROOT   = $MODEL_ROOT"
echo "MERGED_DIT   = $MERGED_DIT"
echo "JSONL        = $JSONL"
echo "OUTDIR       = $OUTDIR"
echo "IMAGE_SIZE   = $IMAGE_SIZE"
echo "STEPS        = $INFER_STEPS"
echo "CFG_SCALE    = $CFG_SCALE"
echo "BATCH_SIZE   = $BATCH_SIZE"
echo "SEED         = $SEED"
echo "================================================"

python3 "$PROJECT_ROOT/sample_t2i_batch.py" \
  --model-root "$MODEL_ROOT" \
  --load-key merge \
  --dit-weight "$MERGED_DIT" \
  --index-file "$JSONL" \
  --results-dir "$OUTDIR" \
  --image-size $IMAGE_SIZE \
  --infer-steps "$INFER_STEPS" \
  --cfg-scale "$CFG_SCALE" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --negative "$NEGATIVE_PROMPT" \
  --infer-mode fa \
  --max-generate-image 500 \


  # Multi-resolution generation
  # --auto-bucket \
  # --bucket-source-json "/data/user/yyh_dk/RamiGen_Gate_Dinov3/dataset/porcelain_military/jsons/porcelain_mt.json" \
  # --bucket-min-count 1 \
  # --bucket-area-weight 0.05

echo
echo "🎉 Inference finished."
echo "📂 Results saved to: $OUTDIR"
