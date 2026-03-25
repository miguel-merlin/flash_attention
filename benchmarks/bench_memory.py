"""
Memory benchmark for attention variants.

Usage:
    source .venv/bin/activate
    python3 benchmarks/bench_memory.py [--cuda]

Profiles peak memory usage:
  - GPU: torch.cuda.memory_stats() peak allocated bytes
  - CPU: tracemalloc peak RSS

Prints a table of peak memory per variant per config.
"""

from __future__ import annotations

import argparse
import gc
import sys
import os
import tracemalloc

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from flash_attn.attention import VanillaAttention
from benchmarks.utils import make_qkv, format_bytes, report_table

CONFIGS = [
    # (B, H,   N,   d)
    (1,  1,   64,  32),
    (1,  1,  128,  64),
    (1,  4,  256,  64),
    (2,  8,  512,  64),
    (4,  8, 1024,  64),
]


# ---------------------------------------------------------------------------
# Memory measurement helpers
# ---------------------------------------------------------------------------

def _measure_cpu_memory(fn, q, k, v) -> int:
    """Return peak CPU memory used by fn(q, k, v) in bytes (via tracemalloc)."""
    gc.collect()
    tracemalloc.start()
    fn(q, k, v)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _measure_gpu_memory(fn, q, k, v) -> int:
    """Return peak GPU memory allocated by fn(q, k, v) in bytes."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn(q, k, v)
    torch.cuda.synchronize()
    return torch.cuda.memory_stats()["allocated_bytes.all.peak"]


def measure_memory(fn, q, k, v, device: str) -> str:
    try:
        if device == "cuda":
            return format_bytes(_measure_gpu_memory(fn, q, k, v))
        else:
            return format_bytes(_measure_cpu_memory(fn, q, k, v))
    except Exception as e:
        return f"ERR: {e}"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmarks(use_cuda: bool = False) -> None:
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    if use_cuda and not torch.cuda.is_available():
        print("[WARNING] --cuda requested but CUDA not available; falling back to CPU.")

    vanilla = VanillaAttention()

    flash_cpp = None
    if device == "cpu":
        try:
            from flash_attn.attention import FlashAttentionCPP
            flash_cpp = FlashAttentionCPP()
            print("[INFO] FlashAttentionCPP loaded.")
        except RuntimeError as e:
            print(f"[WARNING] FlashAttentionCPP unavailable: {e}")

    flash_cuda = None
    if device == "cuda":
        try:
            from flash_attn.attention import FlashAttentionCUDA
            flash_cuda = FlashAttentionCUDA()
            print("[INFO] FlashAttentionCUDA loaded.")
        except RuntimeError as e:
            print(f"[WARNING] FlashAttentionCUDA unavailable: {e}")

    print(f"\n{'='*60}")
    print(f"Memory Benchmark  |  device={device}")
    print(f"{'='*60}\n")

    results = []
    for (B, H, N, d) in CONFIGS:
        config_str = f"B={B} H={H} N={N} d={d}"
        q, k, v = make_qkv(B, H, N, d, device=device)

        row: dict = {"Config": config_str}
        row["Vanilla"] = measure_memory(vanilla, q, k, v, device)

        if flash_cpp is not None:
            row["FlashCPP"] = measure_memory(flash_cpp, q, k, v, device)
        else:
            row["FlashCPP"] = "N/A"

        if flash_cuda is not None:
            row["FlashCUDA"] = measure_memory(flash_cuda, q, k, v, device)
        else:
            row["FlashCUDA"] = "N/A"

        results.append(row)

    print()
    report_table(results)
    print()

    # Additional per-tensor breakdown for the largest config
    B, H, N, d = CONFIGS[-1]
    q, k, v = make_qkv(B, H, N, d, device=device)
    elem_bytes = q.element_size()
    qkv_bytes  = 3 * q.numel() * elem_bytes
    attn_bytes = B * H * N * N * elem_bytes   # materialised attention matrix

    print("--- Theoretical memory for largest config ---")
    print(f"  QKV tensors:         {format_bytes(qkv_bytes)}")
    print(f"  Attention matrix:    {format_bytes(attn_bytes)}")
    print(f"  Total (vanilla):     {format_bytes(qkv_bytes + attn_bytes)}")
    print(f"  Flash (no attn mat): {format_bytes(qkv_bytes)} (+ small SRAM tiles)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Flash Attention memory benchmark")
    parser.add_argument("--cuda", action="store_true", help="Run on CUDA instead of CPU")
    args = parser.parse_args()
    run_benchmarks(use_cuda=args.cuda)


if __name__ == "__main__":
    main()
