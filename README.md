# RamiGen: Region-aware controllable diffusion with self-adaptive conditional evolution for instance-centric remote sensing object generation

**ISPRS Journal of Photogrammetry and Remote Sensing, 2026**

Yueheng Yang, Bin Wang, Wenyan Ran, Ning Zhou, Minting Zhou, Jindi Wang, and Haigang Sui

[Paper](https://doi.org/10.1016/j.isprsjprs.2026.09.048) · \
[Repository](https://github.com/Bynus5/RamiGen-main-FRSMOD) · \
[Overview](#overview) · \
[Installation](#installation) · \
[Data preparation](#data-preparation) · \
[Inference](#inference) · \
[Citation](#citation) · \

## 📝 TODOs

- [x] Publish the paper in ISPRS Journal of Photogrammetry and Remote Sensing.
- [x] Prepare the initial inference code and RACEM.
- [x] Release pretrained models and weights on Hugging Face.
- [ ] Release the training code.
- [ ] Release the FRSMOD dataset.

The initial release focuses on inference and offline RACEM. Training code and FRSMOD datasets will be released later.

## Overview
<img width="1228" height="1000" alt="Fig 2_small" src="https://github.com/user-attachments/assets/d9305cc3-16e1-4780-8e90-a0fd24c33942" />

RamiGen is a diffusion framework for instance-centric object generation in optical and SAR remote sensing imagery. It uses bounding-box annotations as regional prompt to construct fine-grained spatial conditions, improving object geometry, placement, and appearance in sparse and dense scenes.



[RamiGen architecture]
<img width="1488" height="932" alt="overview" src="https://github.com/user-attachments/assets/a4cdf1d9-328e-4393-9a5c-d82b31330331" />



### Main results

Results reported in the paper, Tables 1–3; these are not measurements from the example commands below.

| Dataset | FID ↓ | KID ↓ | YOLO Score ↑ |
| --- | ---: | ---: | ---: |
| DIOR | 37.586 | 0.011 | 0.3212 |
| FRSMOD | 45.761 | 0.008 | 0.7319 |
| SARDet-100K | 55.738 | 0.034 | 0.2371 |

<img width="886" height="761" alt="图片1" src="https://github.com/user-attachments/assets/44e30cfe-c8a4-4975-a802-b363d8375551" />


## Installation

Run all commands from the project root. The tested local stack is Ubuntu 20.04, Python 3.10, PyTorch 2.8.0 + TorchVision 0.23.0 (`cu128`), and Paddle GPU 2.6.2 for HED. `requirements.txt` pins the inference/RACEM dependencies; `environment.yml` uses the same file.

```bash
git clone https://github.com/Bynus5/RamiGen-main-FRSMOD.git
cd RamiGen-main-FRSMOD

conda env create -f environment.yml
conda activate RamiGen

# Install after PyTorch; requires a compatible CUDA Toolkit (nvcc) and C++ compiler.
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip check
```

For an existing Python 3.10 environment, activate it and run `python -m pip install -r requirements.txt` instead of creating another environment. The example inference command uses FlashAttention (`--infer-mode fa`); `--infer-mode torch` does not require the FlashAttention installation.

Keep PyTorch's CUDA 12.8 libraries unchanged; HED uses Paddle's CUDA 11.8 build. Their same-process GPU extraction has been tested locally, but other systems should verify their driver/library setup. Do not install both CPU and GPU Paddle packages or replace these pins with a full Windows environment export. The NVIDIA driver and CUDA Toolkit are not installed by the environment files. See the [official PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/) and [FlashAttention build requirements](https://github.com/Dao-AILab/flash-attention#installation-and-features).

DeepSpeed `0.17.6` is included because the shared argument parser imports it even during inference; this does not launch training. Training code remains outside the initial release.

ONNX/TensorRT tools are also excluded from the default installation. Their locally installed versions are listed as optional comments in `requirements.txt`; they are not needed for `--infer-mode fa` or `torch`. TensorRT requires its own platform-specific setup.

## Pretrained weights

The released **RamiGen DIOR backbone and ControlNet checkpoints**, together with the **HunyuanDiT-v1.2 base model, VAE, text encoders, and tokenizers**, are available from the [RamiGen pretrained checkpoints on Hugging Face](https://huggingface.co/Byuns5/RamiGen_Pretrained_checkpoint). You do not need to download the HunyuanDiT base weights separately.

Run the following commands from the project root to download the complete checkpoint repository:

```bash
python -m pip install -U huggingface_hub
hf download Byuns5/RamiGen_Pretrained_checkpoint \
  --local-dir ckpts/HunyuanDiT-v1.2
```

For the **DIOR inference example only**, you can omit the unused `dialoggen/` and `asset/` files to reduce the download size:

```bash
hf download Byuns5/RamiGen_Pretrained_checkpoint \
  --include "t2i/*" "Pretrained_weights/DIOR_weights/*" \
  --local-dir ckpts/HunyuanDiT-v1.2
```

Both commands preserve the original folder hierarchy on Hugging Face. The relevant checkpoint paths are:

```text
ckpts/
├── RamiGen_component/
│   ├── t2i/
│   │   ├── model/pytorch_model_ema.pt
│   │   ├── clip_text_encoder/
│   │   ├── tokenizer/
│   │   ├── mt5/
│   │   └── sdxl-vae-fp16-fix/
│   ├── Pretrained_weights/
│   │   └── DIOR_weights/
│   │       ├── RamiGen_Backbone_DIOR.pt
│   │       └── DIOR_ControlNet_0029300.pt
│   ├── dialoggen/                   # complete repository download only
│   └── asset/                       # complete repository download only
└── racem/                           # offline condition-extraction weights (separate)
    ├── model_hed.pdparams
    ├── sam2.1_hiera_large.pt
    └── dpt_beit_large_512.pt
```

**RACEM checkpoint files:** The current Hugging Face repository contains the HunyuanDiT assets and RamiGen DIOR weights listed above, but its published file tree does not list the three separate RACEM extraction checkpoints. For offline condition preprocessing, place those files in `ckpts/racem/` using the corresponding upstream sources:

| RACEM checkpoint | Source |
| --- | --- |
| `model_hed.pdparams` | [Paddle HED](https://github.com/txyugood/hed), trained HED checkpoint |
| `sam2.1_hiera_large.pt` | [SAM2](https://github.com/facebookresearch/sam2#download-checkpoints), SAM2.1 Hiera-large |
| `dpt_beit_large_512.pt` | [MiDaS](https://github.com/isl-org/MiDaS#setup), BEiT-large-512 |

Edge extraction uses the supplied Paddle HED implementation with scales `0.5 / 1.0 / 1.5`; no original-SAM edge checkpoint is needed. Paddle GPU 2.6.2 runs alongside the existing PyTorch environment; do not install the CPU-only Paddle package at the same time.

Set the native model root for the commands below:

```bash
MODEL_ROOT=ckpts/HunyuanDiT-v1.2
```

The released DIOR backbone and ControlNet checkpoints are located at `"$MODEL_ROOT/Pretrained_weights/DIOR_weights/"` and are referenced directly in the inference example below. Use a matching checkpoint pair and RoPE configuration when replacing them with your own models.


## Data preparation

The supplied DIOR examples use this layout. Match each image with its label and **English** caption by filename stem:

```text
dataset/porcelain_DIOR/
├── images/00001.jpg                 # reference images
├── labels/00001.txt                 # normalized YOLO annotations
├── text/00001.txt                   # English caption for the matching image
├── csvfile/train.csv                # generated image–caption pairs
├── arrows/00000.arrow               # generated image bytes and captions
├── jsons/
│   ├── porcelain.json              # base image index
│   └── porcelain_mt.json           # multi-resolution bucket index
├── conditions/                     # offline RACEM outputs
│   ├── edge/edge_clip/00001.png
│   ├── depth/depth_clip/00001.png
│   └── mask/mask_clip/00001.png
└── inference_jsonl/test.jsonl       # per-image inference records
```

Each label line is `class_id x1 y1 x2 y2 x3 y3 x4 y4` for OBB, or `class_id center_x center_y width height` for HBB; coordinates are normalized to `[0, 1]`. Keep an empty label file for images without objects. Set the RACEM annotation type to `obb` or `hbb` accordingly.

CSV, Arrow, and index JSON files are preparation artifacts, not supplied data. They are unnecessary for fixed-size inference; generate them for later training or to provide multi-resolution bucket metadata.

### 1. Create the CSV

The converter requires **`img_path` and `text_zh`**. Despite its legacy name, `text_zh` contains our **English** captions; the current converter and default loader use this field. Do not use `text_en` alone. Image bytes, MD5, width, and height are added automatically during Arrow conversion; labels and conditions remain separate files.

```csv
img_path,text_zh
/absolute/path/dataset/porcelain_DIOR/images/00001.jpg,"A golf field surrounded by trees, roads, and residential buildings."
```

Build `csvfile/train.csv` directly from `images/` and `text/` (CSV quoting handles commas in captions):

```bash
python - <<'PY'
import csv
from pathlib import Path

data = Path("dataset/porcelain_DIOR").resolve()
images = sorted(p for p in (data / "images").iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
rows = []
for image in images:
    caption = (data / "text" / f"{image.stem}.txt").read_text(encoding="utf-8").strip()
    if not caption:
        raise ValueError(f"Empty caption: {image.stem}")
    rows.append((str(image), caption))
(data / "csvfile").mkdir(parents=True, exist_ok=True)
with (data / "csvfile" / "train.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["img_path", "text_zh"])
    writer.writerows(rows)
PY
```

### 2. Convert to Arrow and build JSON indexes

Run from the project root:

```bash
pip install -e ./IndexKits
mkdir -p dataset/porcelain_DIOR/arrows dataset/porcelain_DIOR/jsons
python ssadit/data_loader/csv2arrow.py \
  dataset/porcelain_DIOR/csvfile/train.csv dataset/porcelain_DIOR/arrows 1

# Base index
idk base -c dataset/yamls/porcelain_dior.yaml \
  -t dataset/porcelain_DIOR/jsons/porcelain.json

# Multi-resolution index from the same Arrow data
idk multireso --src dataset/porcelain_DIOR/jsons/porcelain.json \
  --base-size 1024 --reso-step 64 --min-size 1 \
  -t dataset/porcelain_DIOR/jsons/porcelain_mt.json
```

The final `1` in `csv2arrow.py` is the worker count. The supplied DIOR YAML selects `arrows/*.arrow` and filters width or height below 800 pixels; adjust its source and filters for another dataset or smaller images.

Multi-resolution indexing groups images by aspect ratio into buckets; do not create resized copies or separate Arrow files for every resolution. `--base-size 1024` sets the bucket reference size, not the RoPE base. For later training, use `porcelain_mt.json` with `--multireso --reso-step 64`. For inference, use `inference_jsonl/test.jsonl` as `--index-file`; the bucket index is passed separately via `--bucket-source-json`.

### 3. Offline conditions

Set these values in [RACEM/RACEM_Level_Set.sh](RACEM/RACEM_Level_Set.sh), then run `bash RACEM/RACEM_Level_Set.sh`:

```bash
RACEM_IMAGES="dataset/porcelain_DIOR/images"
RACEM_LABELS="dataset/porcelain_DIOR/labels"
RACEM_OUTPUT="dataset/porcelain_DIOR/conditions"
RACEM_FEATURES=(all)
RACEM_REGION="clip"
RACEM_BBOX_TYPE="obb"
```

The script saves PNG condition maps using the original image stem, e.g. `00001.jpg` → `00001.png`. Choose `(edge)`, `(depth)`, `(mask)`, or `(edge depth)` for selected features.

| `RACEM_REGION` | Edge/depth region |
| --- | --- |
| `full` | Entire image |
| `bbox` | Inside annotated boxes |
| `clip` | Inside the refined mask |

Output dimensions stay unchanged; mask extraction always uses annotated boxes. `clip` needs the mask backend even for edge/depth-only extraction. Changing the region changes directory suffixes (`edge_clip` → `edge_bbox`, etc.); update inference paths accordingly. Use a new output directory when changing extraction settings. Existing SAUGE edge maps are not converted automatically: regenerate conditions in a new directory (e.g. `conditions_hed`) before using HED outputs.

### 4. Online conditions

Online extraction belongs to the ControlNet trainer, **not** the offline script. Training code is planned for a later release; its data settings are:

| Setting | Value |
| --- | --- |
| `RACEM_MODE` | `batch_wise` |
| `RACEM_LABEL_DIR` | `dataset/porcelain_DIOR/labels` |
| `RACEM_REGION` / `RACEM_BBOX_TYPE` | `clip` / `obb` (or `hbb` for HBB labels) |
| `batch_size` | Per-GPU training batch; no separate RACEM batch size |

The trainer reads images/captions from Arrow, aligns labels with resize/crop/flip, then extracts conditions from each current batch before fusion. It does not require precomputed `conditions/` and does not save condition PNGs. Gradient accumulation does not enlarge an extraction batch.

To read offline maps instead, use `RACEM_MODE=preprocess`, `COND_BASE_DIR=dataset/porcelain_DIOR/conditions`, and `SUB_TYPE=clip` matching the directory suffix. The current inference entry always reads saved maps, so prepare offline conditions for inference even if training used online extraction.

## :golf:Inference

Extract conditions for the reference images with RACEM, matching the region setting and dataset-wide class ID set used by your checkpoints so semantic-mask colors stay consistent. Then create `dataset/porcelain_DIOR/inference_jsonl/test.jsonl`. Each line contains a prompt and paths to the three condition maps, resolved from the project root:

```json
{"file_name":"00001","text":"A golf field surrounded by trees, roads, and residential buildings.","orig_width":800,"orig_height":800,"edge_path":"dataset/porcelain_DIOR/conditions/edge/edge_clip/00001.png","depth_path":"dataset/porcelain_DIOR/conditions/depth/depth_clip/00001.png","mask_path":"dataset/porcelain_DIOR/conditions/mask/mask_clip/00001.png"}
```

Use the paired RamiGen checkpoints prepared above. The example assumes a `base512` checkpoint pair; match the RoPE setting and condition configuration to your models:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python sample_controlnet_multi.py \
  --model-root "$MODEL_ROOT" \
  --dit-weight "$MODEL_ROOT/Pretrained_weights/DIOR_weights/RamiGen_Backbone_DIOR.pt" \
  --controlnet-weight "$MODEL_ROOT/Pretrained_weights/DIOR_weights/DIOR_ControlNet_0029300.pt" \
  --use-condition-fusion --condition-fusion-in-channels 7 \
  --fusion-control-types edge depth mask --control-weight 1.0 \
  --index-file dataset/porcelain_DIOR/inference_jsonl/test.jsonl --results-dir results/inference/DIOR \
  --image-size 1024 1024 --rope-img base512 --rope-real \
  --sampler dpmms --infer-steps 30 --cfg-scale 6.5 --batch-size 1 \
  --seed 42 --infer-mode fa --no-enhance
```

For multi-resolution inference, add `--auto-bucket --bucket-source-json dataset/porcelain_DIOR/jsons/porcelain_mt.json --bucket-min-count 1`. Each JSONL record needs its actual `orig_width` and `orig_height`; otherwise it falls back to `--image-size`. Captions go in the JSONL `text` field, not `text_zh`.

Generated PNG images and `meta.jsonl` are saved under `results/inference/DIOR/`. Fusion uses **one** ControlNet; do not combine it with `--multi-control`. Text-only batch generation is available via `sample_t2i_batch.py`. Full argument help:

```bash
python RACEM/RACEM_Level_Set.py --help
PYTHONPATH=. python sample_controlnet_multi.py --help
```

## :sunrise:Citation

If you use RamiGen in your research, please cite:

```bibtex
@article{yang2026ramigen,
  title   = {RamiGen: region-aware controllable diffusion with self-adaptive conditional evolution for instance-centric remote sensing object generation},
  author  = {Yang, Yueheng and Wang, Bin and Ran, Wenyan and Zhou, Ning and Zhou, Minting and Wang, Jindi and Sui, Haigang},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {242},
  pages   = {1039--1066},
  year    = {2026},
  doi     = {10.1016/j.isprsjprs.2026.09.048}
}
```

## :airplane:Acknowledgements

The implementation builds on [HunyuanDiT](https://github.com/Tencent-Hunyuan/HunyuanDiT), [Paddle HED](https://github.com/txyugood/hed), [SAM2](https://github.com/facebookresearch/sam2), and [MiDaS](https://github.com/isl-org/MiDaS). Preserve upstream licenses and follow each model's usage terms.

Related remote-sensing generation projects: [OF-Diff](https://github.com/VisionXLab/OF-Diff), [AeroGen](https://github.com/Sonettoo/AeroGen), and [MMDiff](https://github.com/XinR-Tang/MMDiff).


