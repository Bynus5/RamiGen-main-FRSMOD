
## Environment Setup

RamiGen is designed for NVIDIA GPUs. The recommended environment is as follows:

| Component | Version |
|-----------|---------|
| GPU | NVIDIA RTX 3090 (24 GB) |
| OS | Linux |
| Python | 3.11 |
| CUDA | 12.6 |
| PyTorch | 2.7.1 |
| FlashAttention | 2.8.3 |

### 1. Create Environment

```bash
conda create -n RamiGen python=3.11 -y
conda activate RamiGen
```

### 2. Install PyTorch

Install PyTorch with CUDA 12.6 support:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install Dependencies

Clone the repository and install the required dependencies:

```bash
git clone <RamiGen-repository-URL>
cd RamiGen

pip install -r requirements.txt
```

**Note:** PyTorch and FlashAttention are installed separately and should not be included in `requirements.txt`.

### 4. Install FlashAttention

FlashAttention requires CUDA Toolkit 12.6 and a compatible C++ compiler. Ensure that `nvcc` is available before installation.

```bash
# Configure CUDA Toolkit (adjust the path if necessary)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH

# Build for RTX 3090 (sm_86)
export FLASH_ATTN_CUDA_ARCHS="86"
export TORCH_CUDA_ARCH_LIST="8.6"

# Build FlashAttention from source
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4

pip install flash-attn==2.8.3 \
    --no-build-isolation \
    --no-deps
```

Alternatively, install a prebuilt wheel compatible with the installed PyTorch, Python, CUDA, and system GLIBC versions.

### 5. Verify Installation

Run the following command to verify PyTorch and FlashAttention:

```bash
python - <<'PY'
import torch
from flash_attn import flash_attn_func

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

q = torch.randn(2, 128, 4, 64, device="cuda",
                dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

out = flash_attn_func(q, k, v)
out.float().square().mean().backward()

torch.cuda.synchronize()

assert all(torch.isfinite(x).all() for x in
           [out, q.grad, k.grad, v.grad])

print("FlashAttention forward/backward: PASS")
PY
```
