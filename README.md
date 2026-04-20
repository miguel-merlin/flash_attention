# Flash Attention

A custom PyTorch CUDA extension for Flash Attention — designed for experimentation with tiled, memory-efficient attention kernels. Includes vanilla and Flash Attention variants, a full autograd backward, and benchmarking / profiling infrastructure.

## Project Layout

```text
flash_attention/
├── flash_attn/
│   ├── __init__.py          # Package entry-point; lazy _C extension loading
│   ├── attention.py         # VanillaAttention, Flash*, NativeVanilla*; optional is_causal
│   ├── ops.py               # Fake/autograd + flash_attention_forward / paged_attention_forward
│   ├── reference.py         # CPU reference implementations (torch ops + manual loops)
│   └── csrc/
│       ├── flash_attention.cpp   # flash_attention + paged_attention TORCH_LIBRARY (CPU + CUDA decls)
│       ├── vanilla_attention.cpp
│       ├── paged_attention.cpp   # CPU paged decode attention + op registration
│       └── cuda/
│           ├── flash_attention.cu
│           ├── vanilla_attention.cu
│           └── paged_attention.cu
├── benchmarks/
│   ├── bench_time.py            # Latency (dense + optional paged op); --cuda
│   ├── bench_memory.py          # Peak memory (dense + optional paged); --cuda
│   ├── bench_tps.py             # End-to-end GPT-2 generation TPS sanity (std vs FlashAttentionCUDA patch)
│   ├── bench_gpt2_prefill.py    # Prefill-only GPT-2 benchmark (full attention over N tokens) — FlashAttention target
│   ├── bench_gpt2_decode.py     # Decode-only GPT-2 benchmark (contiguous KV cache baseline) — PagedAttention target
│   ├── bench_paged_tps.py       # paged_attention_forward throughput (synthetic KV; not LM generate)
│   └── utils.py                 # make_qkv(), make_paged_attention_inputs(), format_bytes(), report_table()
├── tests/
│   ├── test_attention.py    # Correctness + paged CPU/CUDA vs reference
│   └── test_reference.py    # Quick sanity: torch-ops ref vs manual loops
├── llm.py                   # HF: GPT-2 generate with / without patched FlashAttentionCUDA
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

Benchmarks are split by *which phase of inference* they measure, because
FlashAttention and PagedAttention target different phases:

| Phase   | Bottleneck                    | Benchmark script                     | Kernel under test    |
| ------- | ----------------------------- | ------------------------------------ | -------------------- |
| Prefill | Full N×N attention over prompt | `benchmarks/bench_gpt2_prefill.py`   | **FlashAttention**   |
| Decode  | Single-token attention over KV cache | `benchmarks/bench_gpt2_decode.py` | **PagedAttention** (baseline today; paged integration pending) |
| End-to-end | Prefill + repeated decode (both mixed) | `benchmarks/bench_tps.py`      | Full HF `model.generate(...)` sanity |

End-to-end TPS (`bench_tps.py`) is a sanity check only — it does not isolate
attention-kernel effects, so small deltas there do not imply FlashAttention
is (or isn't) working. Use the prefill and decode scripts for honest numbers.

```bash
source .venv/bin/activate

# --- Low-level kernel benchmarks ------------------------------------------
# Latency (CPU / GPU)
python3 benchmarks/bench_time.py
python3 benchmarks/bench_time.py --cuda

# Memory (CPU / GPU)
python3 benchmarks/bench_memory.py
python3 benchmarks/bench_memory.py --cuda

# Paged attention throughput — synthetic paged KV; median latency (CPU or GPU)
python3 benchmarks/bench_paged_tps.py
python3 benchmarks/bench_paged_tps.py --cuda --b 4 --n 512 --iters 200

# --- GPT-2 phase benchmarks -----------------------------------------------
# End-to-end GPT-2 generation TPS sanity (std vs FlashAttentionCUDA patch).
#   --print-output also prints the generated text for correctness spot-check.
python3 benchmarks/bench_tps.py --tokens 50 --iters 5 --print-output

# Prefill-only: single forward pass over the full prompt (full attention).
#   This is where FlashAttention is expected to help.
python3 benchmarks/bench_gpt2_prefill.py --lengths 32,64,128,256,512 --iters 20

# Decode-only: one-token forward with an existing KV cache (fixed length).
#   Baseline for comparing against the future PagedAttention decode integration.
python3 benchmarks/bench_gpt2_decode.py --prompt-len 128 --iters 100
python3 benchmarks/bench_gpt2_decode.py --prompt-len 512 --iters 100

# Decode-only, growing cache (simulates real generation loop).
python3 benchmarks/bench_gpt2_decode.py --prompt-len 128 --iters 50 --grow-cache
```

CPU safety: all three GPT-2 scripts run on CPU without CUDA. `bench_tps.py`
and `bench_gpt2_prefill.py` will skip the `FlashAttentionCUDA` patched run
and clearly report `N/A`; `bench_gpt2_decode.py` does not use any custom
CUDA kernel and runs the baseline on CPU directly.

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
