"""
Benchmark the manual GPT-2 Flash-prefill + Paged-decode prototype.

This times ``experiments.gpt2_flash_paged_generate.generate`` end-to-end for a
fixed prompt and number of new tokens, for one or both modes:

  * ``torch``       : manual GPT-2 forward with the PyTorch attention reference.
  * ``flash-paged`` : FlashAttention prefill + PagedAttention v2 decode.

Important caveat
----------------
This benchmark includes significant Python overhead (per-layer, per-token
Python loops). It is *not* expected to beat Hugging Face ``generate()`` in raw
tokens/sec — it exists to validate correctness end-to-end and to give a rough
feeling for the custom kernel vs. the reference path under identical conditions.

Usage
-----
    # Default (compare both modes, 20 tokens, 5 iters, warmup=1)
    python3 benchmarks/bench_gpt2_flash_paged.py

    # Compare mode with a longer generation
    python3 benchmarks/bench_gpt2_flash_paged.py --tokens 32 --iters 3 --mode compare

    # Time the pure-PyTorch reference only (works on CPU too)
    python3 benchmarks/bench_gpt2_flash_paged.py --mode torch --tokens 20 --iters 3

    # Time the custom kernel path only (requires CUDA + built extension)
    python3 benchmarks/bench_gpt2_flash_paged.py --mode flash-paged --tokens 20 --iters 3

    # Full control over prompt + cache geometry
    python3 benchmarks/bench_gpt2_flash_paged.py \
        --prompt "Once upon a time" --tokens 50 --iters 5 --warmup 2 \
        --max-seq-len 2048 --page-size 32 --device cuda --mode compare
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import List, Tuple

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.gpt2_flash_paged_generate import generate  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench_mode(
    *,
    model,
    tokenizer,
    prompt: str,
    tokens: int,
    mode: str,
    device: torch.device,
    iters: int,
    warmup: int,
    max_seq_len: int,
    page_size: int,
) -> Tuple[List[float], str]:
    """
    Returns (per-iter seconds list, last decoded text). The caller can compute
    median / mean from the list.
    """
    # Warmup
    for _ in range(warmup):
        _sync(device)
        generate(
            model=model, tokenizer=tokenizer, prompt=prompt, tokens=tokens,
            mode=mode, max_seq_len=max_seq_len, page_size=page_size, device=device,
        )
        _sync(device)

    times: List[float] = []
    last_text = ""
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        _ids, last_text = generate(
            model=model, tokenizer=tokenizer, prompt=prompt, tokens=tokens,
            mode=mode, max_seq_len=max_seq_len, page_size=page_size, device=device,
        )
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times, last_text


def _report(name: str, times: List[float], tokens: int) -> None:
    med = statistics.median(times)
    mean = statistics.fmean(times)
    tps = tokens / med if med > 0 else float("inf")
    print(
        f"[{name}] iters={len(times)}  tokens={tokens}  "
        f"median={med*1e3:.1f} ms  mean={mean*1e3:.1f} ms  "
        f"tok/s(median) = {tps:.2f}"
    )


def main() -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Benchmark the manual GPT-2 Flash-prefill + Paged-decode prototype."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The Hugging Face open source models are",
        help="Input prompt text for every timed iteration (default: a short canned prompt).",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=20,
        help="Number of new tokens to generate per iteration (default: 20).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Timed iterations per mode after warmup (default: 5).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations per mode, not timed (default: 1).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Paged KV cache capacity per batch row; must be >= prompt_len + tokens (default: 1024).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="Tokens per paged KV cache page (default: 16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="torch device string (default: 'cuda' if available else 'cpu').",
    )
    parser.add_argument(
        "--mode",
        choices=["torch", "flash-paged", "compare"],
        default="compare",
        help=(
            "Which generation path to benchmark: 'torch' = pure PyTorch reference, "
            "'flash-paged' = FlashAttentionCUDA prefill + paged_attention_v2 decode, "
            "'compare' = run both back to back (default)."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    print(f"prompt: {args.prompt!r}")
    print(
        f"tokens: {args.tokens}  iters: {args.iters}  warmup: {args.warmup}  "
        f"max_seq_len: {args.max_seq_len}  page_size: {args.page_size}"
    )
    print(
        "Caveat: Manual prototype timing includes Python overhead and is not "
        "expected to beat Hugging Face generate."
    )
    print()

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    except Exception as e:
        print(f"[ERROR] Could not load GPT-2: {e}")
        sys.exit(1)

    modes = ["torch", "flash-paged"] if args.mode == "compare" else [args.mode]

    results = {}
    for mode in modes:
        try:
            times, text = _bench_mode(
                model=model, tokenizer=tokenizer,
                prompt=args.prompt, tokens=args.tokens, mode=mode, device=device,
                iters=args.iters, warmup=args.warmup,
                max_seq_len=args.max_seq_len, page_size=args.page_size,
            )
        except RuntimeError as e:
            msg = str(e)
            if "flash-paged mode requires CUDA" in msg or "extension not loaded" in msg:
                print(f"[{mode}] SKIPPED: {msg}")
                continue
            raise
        _report(mode, times, args.tokens)
        print(f"[{mode}] last text: {text!r}")
        results[mode] = statistics.median(times)
        print()

    if "torch" in results and "flash-paged" in results:
        t_torch = results["torch"]
        t_fp = results["flash-paged"]
        if t_fp > 0:
            print(f"speedup flash-paged vs torch (median latency): {t_torch / t_fp:.2f}x")


if __name__ == "__main__":
    main()
