# Flash Attention (CUDA Extension Prototype)

PyTorch CUDA extension prototype for Flash Attention forward pass.

## Repository Layout

```text
flash_attn/
  setup.py                 # CUDA extension build script
  flash_attn.py            # PyTorch autograd wrapper
  csrc/
    flash_attn.cpp         # PyBind / extension entrypoint
    flash_attn.cu          # CUDA forward implementation (WIP)
requirements.txt
```

## Prerequisites
- NVIDIA GPU
- CUDA toolkit installed (`nvcc` available in `PATH`)
- `CUDA_HOME` set (example: `/usr/local/cuda`)
- Python with PyTorch installed

## Setup

1. Create and activate a virtual environment.
2. Install PyTorch first.
3. Build/install the extension in editable mode without build isolation:

```bash
pip install torch
pip install -e flash_attn --no-build-isolation
```

## Usage

```python
import torch
from flash_attn.flash_attn import flash_attention

Q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float32).contiguous()
K = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float32).contiguous()
V = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float32).contiguous()

O = flash_attention(Q, K, V)
print(O.shape)
```

## Troubleshooting

- `CUDA_HOME is not set`:
  - `export CUDA_HOME=/usr/local/cuda`
- `nvcc was not found`:
  - install CUDA toolkit and add `bin/` to `PATH`
- PyTorch import/build errors:
  - install torch first, then reinstall with:
    - `pip install -e flash_attn --no-build-isolation`
