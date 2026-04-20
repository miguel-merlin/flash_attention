"""
Decode-only GPT-2 benchmark (Hugging Face contiguous KV cache baseline).

This script isolates the *decode* phase: given an already-populated KV cache
of length ``prompt_len``, how fast can GPT-2 produce one additional token?

This is the regime where PagedAttention is expected to help, so this file
serves as the honest contiguous-KV baseline we will later compare against our
paged-attention decode integration. It intentionally does NOT touch any
custom CUDA kernel.

Two modes:
    * fixed-cache (default): every timed iteration runs decode against the
      *same* initial cache of length ``prompt_len``. Latency is stable and
      comparable across iters.
    * --grow-cache: the KV cache grows by one token per iter, simulating a
      real generation loop. Latency will slowly increase with cache length.

Usage:
    python3 benchmarks/bench_gpt2_decode.py --prompt-len 128 --iters 100
    python3 benchmarks/bench_gpt2_decode.py --prompt-len 512 --iters 100
    python3 benchmarks/bench_gpt2_decode.py --prompt-len 128 --iters 50 --grow-cache
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from benchmarks.utils import format_time_us


def sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_input_ids(tokenizer, N: int, device: torch.device) -> torch.Tensor:
    """Build a deterministic (1, N) input_ids tensor."""
    base = "The quick brown fox jumps over the lazy dog. " * 64
    ids = tokenizer(base, return_tensors="pt").input_ids[0]
    if ids.numel() < N:
        repeat = (N + ids.numel() - 1) // ids.numel()
        ids = ids.repeat(repeat)
    ids = ids[:N].unsqueeze(0).contiguous().to(device)
    assert ids.shape == (1, N)
    return ids


def _to_legacy(past):
    """
    Normalize Hugging Face cache objects to a legacy tuple-of-tuples of tensors.

    Passing a legacy tuple back into ``model(...)`` creates a fresh internal
    cache object each call, so our stored ``past`` is not mutated in place —
    which is exactly what we want for fixed-cache timing.
    """
    if past is None:
        return None
    if isinstance(past, tuple):
        return past
    to_legacy = getattr(past, "to_legacy_cache", None)
    if callable(to_legacy):
        return to_legacy()
    return past


def median_time_decode(
    model,
    next_token: torch.Tensor,
    initial_past,
    warmup: int,
    iters: int,
    grow_cache: bool,
) -> float:
    """
    Time one-token decode ``warmup + iters`` times and return median seconds.

    When ``grow_cache`` is False (default), every iteration feeds the *same*
    ``initial_past`` cache, so the measured attention length stays fixed at
    ``prompt_len``. When True, we update ``past`` with the cache returned by
    the model, so cache length grows by 1 per iter (real generation).
    """
    with torch.no_grad():
        past = initial_past
        for _ in range(warmup):
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            if grow_cache:
                past = _to_legacy(out.past_key_values)
        sync_if_cuda()

        times: list[float] = []
        past = initial_past if not grow_cache else past
        for _ in range(iters):
            sync_if_cuda()
            start = time.perf_counter()
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            sync_if_cuda()
            times.append(time.perf_counter() - start)
            if grow_cache:
                past = _to_legacy(out.past_key_values)

    times.sort()
    return times[len(times) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode-only GPT-2 benchmark (contiguous KV cache baseline)"
    )
    parser.add_argument("--prompt-len", type=int, default=128, help="Prefill/cache length")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations")
    parser.add_argument(
        "--grow-cache",
        action="store_true",
        help="Grow the KV cache each iter (simulates real generation).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "grow-cache" if args.grow_cache else "fixed-cache"

    print("=" * 72)
    print("Decode-only GPT-2 benchmark (Hugging Face contiguous KV cache baseline)")
    print("=" * 72)
    print(f"Device:       {device}")
    print(f"Prompt/cache: {args.prompt_len}")
    print(f"Mode:         {mode}")
    print(f"Warmup/iters: {args.warmup}/{args.iters}")
    if device.type == "cpu":
        print(
            "WARNING: running on CPU. Decode latency will be slow and is not "
            "representative of GPU numbers."
        )

    print("\nLoading GPT-2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    # Prefill once to populate KV cache
    input_ids = make_input_ids(tokenizer, args.prompt_len, device)
    with torch.no_grad():
        sync_if_cuda()
        prefill_out = model(input_ids=input_ids, use_cache=True)
        sync_if_cuda()
    initial_past = _to_legacy(prefill_out.past_key_values)

    # Next-token input (shape (1, 1)); use EOS as a deterministic valid token id
    next_token = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long, device=device)

    # Time decode
    median = median_time_decode(
        model,
        next_token,
        initial_past,
        warmup=args.warmup,
        iters=args.iters,
        grow_cache=args.grow_cache,
    )
    tps = 1.0 / median if median > 0 else float("inf")

    print("\nResults:")
    print(f"  device            : {device}")
    print(f"  cache length      : {args.prompt_len} ({'growing' if args.grow_cache else 'fixed'})")
    print(f"  median decode time: {format_time_us(median)}")
    print(f"  decode tokens/sec : {tps:.2f}")

    print(
        "\nNote: this benchmark measures one-token decode latency with the "
        "standard Hugging Face contiguous KV cache. No custom CUDA attention "
        "kernel is used; this is the baseline for the future PagedAttention "
        "decode integration."
    )


if __name__ == "__main__":
    main()
