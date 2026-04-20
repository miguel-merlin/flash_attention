"""
Prefill-only GPT-2 benchmark.

This script isolates the *prefill* phase of GPT-2 — a single forward pass
over a prompt of length N where full (N x N) attention is computed. This
is the regime where FlashAttention is expected to help, so patching the
attention module with our custom CUDA kernel is meaningful here.

We intentionally do NOT call ``model.generate(...)``. That would mix prefill
with repeated single-token decodes and hide any kernel-level delta.

Usage:
    python3 benchmarks/bench_gpt2_prefill.py --lengths 32,64,128,256,512 --iters 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from benchmarks.utils import format_time_us


def patch_gpt2_attention(model, custom_attn_module) -> None:
    """Patch every GPT-2 block's ``_attn`` to use the custom attention module."""
    for block in model.transformer.h:
        def custom_attn_forward(self, query, key, value, attention_mask=None, head_mask=None):
            out = custom_attn_module(query, key, value, is_causal=True)
            return out, None
        block.attn._attn = types.MethodType(custom_attn_forward, block.attn)


def sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_input_ids(tokenizer, N: int, device: torch.device) -> torch.Tensor:
    """
    Build a deterministic (1, N) input_ids tensor by repeating a short prompt
    and truncating/padding to exactly N tokens.
    """
    base = "The quick brown fox jumps over the lazy dog. " * 64
    ids = tokenizer(base, return_tensors="pt").input_ids[0]
    if ids.numel() < N:
        repeat = (N + ids.numel() - 1) // ids.numel()
        ids = ids.repeat(repeat)
    ids = ids[:N].unsqueeze(0).contiguous().to(device)
    assert ids.shape == (1, N), f"expected (1, {N}), got {tuple(ids.shape)}"
    return ids


def median_time(model, input_ids: torch.Tensor, warmup: int, iters: int) -> float:
    """Run prefill ``warmup + iters`` times; return median wall-clock time in seconds."""
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids=input_ids, use_cache=True)
        sync_if_cuda()

        times: list[float] = []
        for _ in range(iters):
            sync_if_cuda()
            start = time.perf_counter()
            _ = model(input_ids=input_ids, use_cache=True)
            sync_if_cuda()
            times.append(time.perf_counter() - start)

    times.sort()
    return times[len(times) // 2]


def parse_lengths(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefill-only GPT-2 benchmark")
    parser.add_argument(
        "--lengths",
        type=str,
        default="32,64,128,256,512",
        help="Comma-separated prompt lengths",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    args = parser.parse_args()

    lengths = parse_lengths(args.lengths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("Prefill-only GPT-2 benchmark (single forward pass over full prompt)")
    print("=" * 72)
    print(f"Device: {device}")
    if device.type == "cpu":
        print(
            "WARNING: running on CPU. Standard prefill will run but may be slow; "
            "the FlashAttentionCUDA patched benchmark will be skipped."
        )

    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Standard GPT-2 (fresh, unpatched model)
    print("Loading standard GPT-2 model...")
    std_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    std_model.eval()

    standard_latencies: dict[int, float] = {}
    for N in lengths:
        input_ids = make_input_ids(tokenizer, N, device)
        t = median_time(std_model, input_ids, warmup=args.warmup, iters=args.iters)
        standard_latencies[N] = t
        print(f"  standard  N={N:<5d} median={format_time_us(t)}")

    del std_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Patched GPT-2 with FlashAttentionCUDA (fresh model so the standard run is untouched)
    patched_latencies: dict[int, float | None] = {N: None for N in lengths}
    if device.type == "cuda":
        try:
            from flash_attn.attention import FlashAttentionCUDA

            print("Loading patched GPT-2 model (FlashAttentionCUDA)...")
            patched_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
            patched_model.eval()
            custom_attn = FlashAttentionCUDA(device=device)
            patch_gpt2_attention(patched_model, custom_attn)

            for N in lengths:
                input_ids = make_input_ids(tokenizer, N, device)
                t = median_time(patched_model, input_ids, warmup=args.warmup, iters=args.iters)
                patched_latencies[N] = t
                print(f"  patched   N={N:<5d} median={format_time_us(t)}")
        except Exception as e:
            print(f"Could not benchmark patched FlashAttentionCUDA: {e}")
    else:
        print("[ Skipping FlashAttentionCUDA patched prefill — CUDA not available ]")

    # Results table
    print("\nResults (median latency per prefill):")
    header = f"{'N':>6}  {'standard':>14}  {'flash_patched':>16}  {'speedup':>10}"
    print(header)
    print("-" * len(header))
    for N in lengths:
        std_t = standard_latencies[N]
        flash_t = patched_latencies[N]
        if flash_t is None:
            flash_str = "N/A"
            speedup_str = "N/A"
        else:
            flash_str = format_time_us(flash_t)
            speedup_str = f"{std_t / flash_t:.2f}x" if flash_t > 0 else "inf"
        print(f"{N:>6}  {format_time_us(std_t):>14}  {flash_str:>16}  {speedup_str:>10}")

    print(
        "\nNote: this benchmark measures only the prefill forward pass "
        "(full attention over N tokens), not generation."
    )


if __name__ == "__main__":
    main()
