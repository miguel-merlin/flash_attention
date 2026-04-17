# Flash Attention

A custom PyTorch CUDA extension for Flash Attention — designed for experimentation with tiled, memory-efficient attention kernels. Includes vanilla and Flash Attention variants, a full autograd backward, and benchmarking / profiling infrastructure.

## Project Layout

```text
flash_attention/
├── flash_attn/
│   ├── __init__.py          # Package entry-point; lazy _C extension loading
│   ├── attention.py         # nn.Module variants: VanillaAttention, FlashAttentionCPP, FlashAttentionCUDA
│   ├── ops.py               # register_fake, register_autograd, flash_attention_forward()
│   ├── reference.py         # CPU reference implementations (torch ops + manual loops)
│   └── csrc/
│       ├── flash_attention.cpp          # TORCH_LIBRARY registration; CPU forward + backward stubs
│       └── cuda/
│           └── flash_attention.cu       # CUDA forward + backward kernel stubs (TODO: implement)
├── benchmarks/
│   ├── bench_time.py        # Latency benchmarks (torch.utils.benchmark); --cuda flag
│   ├── bench_memory.py      # Peak memory profiling (tracemalloc / cuda.memory_stats)
│   ├── bench_tps.py         # Tokens per second benchmark evaluating text generation limits
│   └── utils.py             # make_qkv(), format_bytes(), report_table()
├── tests/
│   └── test_attention.py    # Correctness tests + gradcheck + finite-difference backward check
├── test_reference.py        # Quick sanity check: torch-ops ref vs manual loops
├── llm.py                   # HF integration: monkey-patches GPT-2 to use custom causal attention
├── setup.py                 # CUDA extension build config
├── pyproject.toml
└── requirements.txt
```

## Attention Variants

| Class                | Device     | Memory          | Description                                                     |
| -------------------- | ---------- | --------------- | --------------------------------------------------------------- |
| `VanillaAttention`   | CPU & CUDA | O(N²)           | Pure PyTorch; fully differentiable; baseline                    |
| `FlashAttentionCPP`  | CPU only   | O(N²)           | C++ CPU kernel via `TORCH_LIBRARY`; requires compiled extension |
| `FlashAttentionCUDA` | CUDA       | O(N) _(target)_ | CUDA kernel stub — fill in `flash_attention.cu`                 |

All variants take `(B, H, N, d)` tensors and return `(B, H, N, d)`.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA toolkit (`nvcc` in `PATH`, `CUDA_HOME` set)
- PyTorch ≥ 2.0

## Setup

```bash
# 1. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build and install the CUDA extension (required for FlashAttentionCPP / CUDA)
pip install -e . --no-build-isolation
```

> **Note:** The package imports cleanly without the compiled extension. `VanillaAttention` and the reference implementations always work. `FlashAttentionCPP` / `FlashAttentionCUDA` raise a clear `RuntimeError` with build instructions if the extension is missing.

## Usage

```python
import torch
from flash_attn import VanillaAttention, FlashAttentionCPP, FlashAttentionCUDA

B, H, N, d = 2, 8, 512, 64
q = torch.randn(B, H, N, d)
k = torch.randn(B, H, N, d)
v = torch.randn(B, H, N, d)

# Pure-PyTorch baseline (no extension needed)
attn = VanillaAttention()
out = attn(q, k, v)          # (B, H, N, d)

# C++ CPU kernel (requires pip install -e .)
attn_cpp = FlashAttentionCPP()
out_cpp  = attn_cpp(q, k, v)

# CUDA kernel (requires pip install -e . and a GPU)
q_gpu = q.cuda()
attn_cuda = FlashAttentionCUDA()
out_cuda  = attn_cuda(q_gpu, k.cuda(), v.cuda())
```

### Autograd

All variants support standard PyTorch autograd:

```python
q = torch.randn(B, H, N, d, requires_grad=True)
k = torch.randn(B, H, N, d, requires_grad=True)
v = torch.randn(B, H, N, d, requires_grad=True)

out = VanillaAttention()(q, k, v)
out.sum().backward()   # dq, dk, dv are populated
```

The `FlashAttentionCPP` / `FlashAttentionCUDA` paths use an analytically correct Python backward by default. Replace `_backward` in `flash_attn/ops.py` with your CUDA backward kernel once implemented.

## Running Tests

```bash
source .venv/bin/activate

# Run the full test suite (reference parity + attention correctness + backward)
python3 -m unittest discover -s tests -v
```

`FlashAttentionCPP` tests auto-skip until the extension is compiled.

## Running Benchmarks

```bash
source .venv/bin/activate

# Latency (CPU)
python3 benchmarks/bench_time.py

# Latency (GPU)
python3 benchmarks/bench_time.py --cuda

# Memory (CPU)
python3 benchmarks/bench_memory.py

# Memory (GPU)
python3 benchmarks/bench_memory.py --cuda

# Tokens Per Second (TPS) Evaluation for text generation (GPU)
python3 benchmarks/bench_tps.py
```

## Implementing the CUDA Kernel

1. **Forward** — edit `flash_attn/csrc/cuda/flash_attention.cu` → `flash_attention_forward_kernel`
   - Tile Q, K, V into SRAM blocks of size `Br × d` and `Bc × d`
   - Compute softmax incrementally (online normalisation à la FlashAttention paper)
   - Write output without materialising the full N×N attention matrix

2. **Backward** — edit `flash_attention_backward_kernel` in the same file
   - Reload tiles from HBM; recompute S and P; accumulate dQ, dK, dV in SRAM

3. **Compile** — `pip install -e . --no-build-isolation`

4. **Validate** — `python3 tests/test_attention.py` (FlashAttentionCPP tests will now run)

## Troubleshooting

| Error                                                      | Fix                                                                  |
| ---------------------------------------------------------- | -------------------------------------------------------------------- |
| `CUDA_HOME is not set`                                     | `export CUDA_HOME=/usr/local/cuda`                                   |
| `nvcc was not found`                                       | Install CUDA toolkit; add `bin/` to `PATH`                           |
| `operator flash_attention::flash_attention does not exist` | Extension not compiled — run `pip install -e . --no-build-isolation` |
| PyTorch import errors during build                         | Install torch first, then rebuild                                    |
