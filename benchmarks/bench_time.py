"""
Timing benchmark for attention variants.

Usage:
    source .venv/bin/activate
    python3 benchmarks/bench_time.py [--cuda]

Measures wall-clock latency using torch.utils.benchmark.Timer with warmup.
Prints a side-by-side table for each (B, H, N, d) configuration.
"""

from __future__ import annotations

import argparse
import sys
import os

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.utils.benchmark as benchmark

from flash_attn.attention import VanillaAttention
from benchmarks.utils import make_qkv, format_time_us, report_table

# ---------------------------------------------------------------------------
# Benchmark configurations — add / remove as needed
# ---------------------------------------------------------------------------
CONFIGS = [
    # (B, H,   N,   d)
    (1,  1,   64,  32),
    (1,  1,  128,  64),
    (1,  4,  256,  64),
    (2,  8,  512,  64),
    (4,  8, 1024,  64),
    (2, 16, 2048, 128),
]

NUM_THREADS = 1   # set > 1 to benchmark multi-threaded CPU
WARMUP_ITERS = 5
BENCH_ITERS = 20


def bench_one(fn, label: str, q, k, v, device: str) -> float:
    """Return median wall-clock time in seconds for fn(q, k, v)."""
    timer = benchmark.Timer(
        stmt="fn(q, k, v)",
        globals={"fn": fn, "q": q, "k": k, "v": v},
        num_threads=NUM_THREADS,
        label=label,
    )
    result = timer.timeit(BENCH_ITERS)
    return result.median  # seconds


def run_benchmarks(use_cuda: bool = False) -> None:
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    if use_cuda and not torch.cuda.is_available():
        print("[WARNING] --cuda requested but CUDA not available; falling back to CPU.")

    vanilla = VanillaAttention()

    # Attempt to import FlashAttentionCPP; skip gracefully if extension missing
    flash_cpp = None
    if device == "cpu":
        try:
            from flash_attn.attention import FlashAttentionCPP
            flash_cpp = FlashAttentionCPP()
            print("[INFO] FlashAttentionCPP loaded.")
        except RuntimeError as e:
            print(f"[WARNING] FlashAttentionCPP unavailable: {e}")

    flash_cuda = None
    native_vanilla_cuda = None
    if device == "cuda":
        try:
            from flash_attn.attention import FlashAttentionCUDA
            flash_cuda = FlashAttentionCUDA()
            print("[INFO] FlashAttentionCUDA loaded.")
        except RuntimeError as e:
            print(f"[WARNING] FlashAttentionCUDA unavailable: {e}")
        try:
            from flash_attn.attention import NativeVanillaAttentionCUDA
            native_vanilla_cuda = NativeVanillaAttentionCUDA()
            print("[INFO] NativeVanillaCUDA loaded.")
        except RuntimeError as e:
            print(f"[WARNING] NativeVanillaCUDA unavailable: {e}")

    print(f"\n{'='*60}")
    print(f"Timing Benchmark  |  device={device}  |  warmup={WARMUP_ITERS}  |  iters={BENCH_ITERS}")
    print(f"{'='*60}\n")

    results = []
    for (B, H, N, d) in CONFIGS:
        if device == "cpu" and N >= 2048:
            continue
        config_str = f"B={B} H={H} N={N} d={d}"
        q, k, v = make_qkv(B, H, N, d, device=device)

        row: dict = {"Config": config_str}

        # --- VanillaAttention ---
        t = bench_one(vanilla, "VanillaAttention", q, k, v, device)
        row["Vanilla"] = format_time_us(t)

        # --- FlashAttentionCPP ---
        if flash_cpp is not None:
            try:
                t = bench_one(flash_cpp, "FlashAttentionCPP", q, k, v, device)
                row["FlashCPP"] = format_time_us(t)
            except Exception as e:
                row["FlashCPP"] = f"ERR: {e}"
        else:
            row["FlashCPP"] = "N/A"

        # --- FlashAttentionCUDA ---
        if flash_cuda is not None:
            try:
                t = bench_one(flash_cuda, "FlashAttentionCUDA", q, k, v, device)
                row["FlashCUDA"] = format_time_us(t)
            except Exception as e:
                row["FlashCUDA"] = f"ERR: {e}"
        else:
            row["FlashCUDA"] = "N/A"
            
        # --- NativeVanillaCUDA ---
        if native_vanilla_cuda is not None:
            try:
                t = bench_one(native_vanilla_cuda, "NativeVanillaCUDA", q, k, v, device)
                row["NativeCUDA"] = format_time_us(t)
            except Exception as e:
                row["NativeCUDA"] = f"ERR: {e}"
        else:
            row["NativeCUDA"] = "N/A"

        results.append(row)
        print(f"Finished {config_str}")

    print()
    report_table(results)
    print()


def main():
    parser = argparse.ArgumentParser(description="Flash Attention timing benchmark")
    parser.add_argument("--cuda", action="store_true", help="Run on CUDA instead of CPU")
    args = parser.parse_args()
    run_benchmarks(use_cuda=args.cuda)


if __name__ == "__main__":
    main()
