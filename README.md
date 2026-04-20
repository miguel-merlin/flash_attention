# Flash Attention

A custom PyTorch CUDA extension for Flash Attention — designed for experimentation with tiled, memory-efficient attention kernels. Includes vanilla and Flash Attention variants, a full autograd backward, and benchmarking / profiling infrastructure.

## Project Layout

```text
flash_attention/
├── flash_attn/
│   ├── __init__.py          # Package entry-point; lazy _C extension loading
│   ├── attention.py         # VanillaAttention, Flash*, NativeVanilla*; optional is_causal
│   ├── ops.py               # Fake/autograd + flash_attention_forward / paged_attention{,_v2}_forward
│   ├── reference.py         # CPU reference implementations (torch ops + manual loops)
│   └── csrc/
│       ├── flash_attention.cpp   # flash_attention + paged_attention TORCH_LIBRARY (CPU + CUDA decls)
│       ├── vanilla_attention.cpp
│       ├── paged_attention.cpp   # CPU paged decode attention + op registration (v1 + v2 dispatch)
│       └── cuda/
│           ├── flash_attention.cu
│           ├── vanilla_attention.cu
│           ├── paged_attention.cu
│           └── paged_attention_v2.cu
├── experiments/
│   ├── paged_kv_cache.py              # SimplePagedKVCache: educational paged KV-cache manager (CPU-safe)
│   └── gpt2_flash_paged_generate.py   # Manual GPT-2 with Flash prefill + Paged v2 decode (--mode torch|flash-paged|compare)
├── benchmarks/
│   ├── bench_time.py                  # Latency (dense + optional paged op); --cuda
│   ├── bench_memory.py                # Peak memory (dense + optional paged); --cuda
│   ├── bench_tps.py                   # End-to-end GPT-2 generation TPS sanity (std vs FlashAttentionCUDA patch)
│   ├── bench_gpt2_prefill.py          # Prefill-only GPT-2 benchmark (full attention over N tokens) — FlashAttention target
│   ├── bench_gpt2_decode.py           # Decode-only GPT-2 benchmark (contiguous KV cache baseline) — PagedAttention target
│   ├── bench_paged_tps.py             # paged_attention{,_v2}_forward throughput (synthetic KV; not LM generate)
│   ├── bench_gpt2_flash_paged.py      # Manual prototype: torch vs flash-paged end-to-end generate
│   └── utils.py                       # make_qkv(), make_paged_attention_inputs(), format_bytes(), report_table()
├── tests/
│   ├── test_attention.py                        # Correctness + paged CPU/CUDA vs reference
│   ├── test_reference.py                        # Quick sanity: torch-ops ref vs manual loops
│   └── test_gpt2_paged_attention_decode.py      # Bridge test: real GPT-2 Q/K/V → paged cache → paged_attention_v2 vs torch ref
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

## Manual GPT-2 Flash-prefill + Paged-decode Prototype

The `experiments/` directory contains a minimal, readable GPT-2 generator that
wires our custom kernels to the two phases they're each designed for:

- **Prefill** goes through `FlashAttentionCUDA` (full-attention kernel).
- **Decode** goes through `paged_attention_v2_forward` on top of a
  `SimplePagedKVCache` that matches the kernel's expected memory layout.

This is an *educational* prototype, not a production inference engine: batch
size is 1, decoding is greedy, and each layer is stepped in a Python loop.
It's there to prove end-to-end correctness on real GPT-2 activations before
any optimization.

### `experiments/paged_kv_cache.py` — `SimplePagedKVCache`

A pure-PyTorch paged KV-cache manager (no custom CUDA ops needed, so it works
on CPU). Layout matches what `paged_attention_v2` expects:

```
k_cache[layer]: (P, page_size, H, d)
v_cache[layer]: (P, page_size, H, d)
page_table:     (B, max_pages)   int64
seq_lens:       (B,)             int64
```

Uses a static identity allocation: `page_table[b, lp] = b * max_pages + lp`.

Public API:

- `SimplePagedKVCache(num_layers, num_heads, head_dim, max_seq_len, page_size=16, batch_size=1, device="cuda", dtype=torch.float32)`
- `reset()` — zero caches and `seq_lens`.
- `write_prefill(layer_idx, k, v, batch_idx=0)` — accepts `(B,H,N,d)`, `(B,N,H,d)`, `(H,N,d)` or `(N,H,d)`; sets `seq_lens[batch_idx] = N`.
- `append_decode(layer_idx, k_new, v_new, batch_idx=0, position=None)` — accepts `(B,H,d)`, `(B,H,1,d)` or `(H,d)`. When writing K/V for the same decode step across many layers, pass an explicit `position=` so all layers write to the same token slot and `seq_lens` only advances once per step.
- `get_layer_cache(layer_idx)` → `(k_cache, v_cache, page_table, seq_lens)` — pass straight to the kernel.
- `reconstruct(layer_idx, batch_idx=0)` → `(k, v)` each shape `(1, H, N, d)` — useful for catching page-layout bugs before blaming the kernel.

The file has a CPU self-test under `if __name__ == "__main__"` that round-trips
prefill, exercises multi-layer decode at a shared `position=`, and verifies
alternate input shapes.

```bash
# Pure CPU; no compiled extension needed
python3 experiments/paged_kv_cache.py
# => SimplePagedKVCache CPU self-test: PASS
```

### `experiments/gpt2_flash_paged_generate.py` — Manual GPT-2 Generator

Manually re-implements GPT-2's forward pass so each phase can be routed
through the right kernel:

| Mode          | Prefill attention       | Decode attention                |
| ------------- | ----------------------- | ------------------------------- |
| `torch`       | PyTorch scaled dot-prod | PyTorch scaled dot-prod (grown contiguous cache) |
| `flash-paged` | `FlashAttentionCUDA`    | `paged_attention_v2_forward` on `SimplePagedKVCache` |
| `compare`     | runs `torch` then `flash-paged` and prints both outputs |

```bash
# Pure-PyTorch reference — works on CPU or CUDA
python3 experiments/gpt2_flash_paged_generate.py --mode torch --tokens 20

# Requires CUDA + compiled extension
python3 experiments/gpt2_flash_paged_generate.py --mode flash-paged --tokens 20

# Side-by-side (CUDA)
python3 experiments/gpt2_flash_paged_generate.py --mode compare --tokens 20
```

CLI: `--prompt`, `--tokens`, `--max-seq-len`, `--page-size`, `--device`, `--mode`.

If CUDA or the compiled extension is missing, `flash-paged` prints a clear
message and exits cleanly instead of crashing; `torch` mode always works.

Under greedy decoding, both modes should produce **identical token IDs** on
CUDA — this is the primary end-to-end correctness check for the custom
kernels on a real model.

## Running Tests

```bash
source .venv/bin/activate

# Run the full test suite (reference parity + attention correctness + backward)
python3 -m unittest discover -s tests -v
```

`FlashAttentionCPP` tests auto-skip until the extension is compiled.

### Bridge test: real GPT-2 Q/K/V → paged attention

`tests/test_gpt2_paged_attention_decode.py` is the bridge between synthetic
paged-attention tests and a real model. It:

1. Pulls real Q/K/V out of `GPT2LMHeadModel` block 0 by manually running
   `wte + wpe → ln_1 → c_attn → split_heads`.
2. Writes the K/V into `SimplePagedKVCache` and asserts the reconstruction is
   bitwise-identical (catches layout bugs without blaming the kernel).
3. Compares `paged_attention_v2_forward(q_decode, ...)` against a PyTorch
   decode-attention reference and asserts `allclose(atol=1e-3, rtol=1e-3)`.

```bash
# Runs as a script too; prints device / shapes / errors / PASS|FAIL.
python3 tests/test_gpt2_paged_attention_decode.py
```

On CPU-only machines the reconstruction sub-test still runs; the CUDA-only
sub-test skips cleanly with a clear message.

## Running Benchmarks

Benchmarks are split by *which phase of inference* they measure, because
FlashAttention and PagedAttention target different phases:

| Phase   | Bottleneck                    | Benchmark script                     | Kernel under test    |
| ------- | ----------------------------- | ------------------------------------ | -------------------- |
| Prefill | Full N×N attention over prompt | `benchmarks/bench_gpt2_prefill.py`   | **FlashAttention**   |
| Decode  | Single-token attention over KV cache | `benchmarks/bench_gpt2_decode.py` | **PagedAttention** (HF contiguous-KV baseline) |
| Prefill + Decode (manual) | Flash prefill + Paged decode | `benchmarks/bench_gpt2_flash_paged.py` | **FlashAttention + PagedAttention v2** (Python-loop prototype) |
| End-to-end (HF) | Prefill + repeated decode (both mixed) | `benchmarks/bench_tps.py`      | Full HF `model.generate(...)` sanity |

End-to-end TPS (`bench_tps.py`) is a sanity check only — it does not isolate
attention-kernel effects, so small deltas there do not imply FlashAttention
is (or isn't) working. Use the prefill and decode scripts for honest numbers.
The manual `bench_gpt2_flash_paged.py` isolates both custom kernels at once
but carries Python-loop overhead (see "Manual GPT-2 prototype" above).

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

# --- Manual Flash-prefill + Paged-decode prototype ------------------------
# End-to-end manual GPT-2 generator: times 'torch' vs 'flash-paged' modes
# under identical greedy decoding. Includes Python-loop overhead, so it is
# not expected to beat HF generate; use it to measure kernel-level changes
# against a fixed harness.
python3 benchmarks/bench_gpt2_flash_paged.py --tokens 20 --iters 3 --mode compare
```

CPU safety: all three GPT-2 scripts run on CPU without CUDA. `bench_tps.py`
and `bench_gpt2_prefill.py` will skip the `FlashAttentionCUDA` patched run
and clearly report `N/A`; `bench_gpt2_decode.py` does not use any custom
CUDA kernel and runs the baseline on CPU directly. `bench_gpt2_flash_paged.py`
cleanly skips `flash-paged` on CPU-only machines and still benchmarks
`torch` mode.

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
