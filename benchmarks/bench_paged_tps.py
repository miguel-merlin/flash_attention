"""
Paged attention throughput benchmark (synthetic KV cache).

This is **not** Hugging Face LM tokens/sec (see ``bench_tps.py`` for GPT-2 generation).
It times ``paged_attention_forward`` on inputs from ``make_paged_attention_inputs``:
one decode-style attention step per call over ``N`` cached tokens per batch row.

Usage:
    source .venv/bin/activate
    python3 benchmarks/bench_paged_tps.py
    python3 benchmarks/bench_paged_tps.py --cuda --b 4 --n 512 --iters 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from benchmarks.utils import format_time_us, make_paged_attention_inputs


def bench_paged_forward(
    paged_fn,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    is_causal: bool,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    """
    Returns (median_seconds_per_call, calls_per_second).
    """
    device = q.device
    for _ in range(warmup):
        paged_fn(q, k_cache, v_cache, page_table, seq_lens, is_causal)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        paged_fn(q, k_cache, v_cache, page_table, seq_lens, is_causal)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    return median, 1.0 / median if median > 0 else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paged attention: median latency and throughput (synthetic cache)"
    )
    parser.add_argument("--cuda", action="store_true", help="Run on CUDA (default: CPU)")
    parser.add_argument("--b", type=int, default=1, help="Batch size")
    parser.add_argument("--h", type=int, default=8, help="Number of heads")
    parser.add_argument("--n", type=int, default=128, help="Cached sequence length (keys 0..N-1)")
    parser.add_argument("--d", type=int, default=64, help="Head dimension")
    parser.add_argument("--page-size", type=int, default=16, help="KV page size")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations (after warmup)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--is-causal",
        action="store_true",
        help="Pass is_causal=True (decode-only cache: numerically same as False for this layout)",
    )
    parser.add_argument(
        "--impl",
        choices=["v1", "v2", "both"],
        default="both",
        help="Which paged-attention kernel(s) to benchmark",
    )
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    if args.cuda and not torch.cuda.is_available():
        print("[WARNING] --cuda requested but CUDA not available; using CPU.")
    device = torch.device("cuda" if use_cuda else "cpu")

    try:
        import flash_attn as fa

        if not getattr(fa, "_EXTENSION_LOADED", False):
            raise RuntimeError("flash_attn extension not built (pip install -e .)")
        from flash_attn.ops import paged_attention_forward
        paged_v2_forward = None
        if args.impl in ("v2", "both"):
            try:
                from flash_attn.ops import paged_attention_v2_forward

                paged_v2_forward = paged_attention_v2_forward
            except (ImportError, AttributeError) as e:
                print(f"[WARNING] paged_attention v2 unavailable: {e}")
                if args.impl == "v2":
                    sys.exit(1)
    except Exception as e:
        print(f"[ERROR] paged_attention not available: {e}")
        sys.exit(1)

    impls: list[tuple[str, Callable]] = []
    if args.impl in ("v1", "both"):
        impls.append(("v1", paged_attention_forward))
    if args.impl in ("v2", "both") and paged_v2_forward is not None:
        impls.append(("v2", paged_v2_forward))

    B, H, N, d = args.b, args.h, args.n, args.d
    page_size = args.page_size
    is_causal = args.is_causal

    print(f"Device: {device}")
    print(
        f"Config: B={B}  H={H}  d={d}  cached_tokens={N}  page_size={page_size}  "
        f"is_causal={is_causal}"
    )

    q, k_cache, v_cache, page_table, seq_lens = make_paged_attention_inputs(
        B, H, N, d, page_size=page_size, device=device
    )

    results: list[tuple[str, float, float]] = []
    for name, fn in impls:
        median_s, calls_per_s = bench_paged_forward(
            fn,
            q,
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            is_causal=is_causal,
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append((name, median_s, calls_per_s))

    # One forward = B independent decode queries (one "row" per sequence in batch).
    print()
    header = f"{'impl':<6}  {'latency/fwd':>14}  {'fwd/s':>12}  {'rows/s':>12}"
    print(header)
    print("-" * len(header))
    for name, median_s, calls_per_s in results:
        query_rows_per_s = B / median_s if median_s > 0 else float("inf")
        print(
            f"{name:<6}  {format_time_us(median_s):>14}  "
            f"{calls_per_s:>12,.1f}  {query_rows_per_s:>12,.1f}"
        )

    if len(results) == 2:
        (_, t1, _), (_, t2, _) = results
        if t2 > 0:
            print(f"\nSpeedup v2 vs v1: {t1 / t2:.2f}x  (lower latency is better)")

    print(
        "\nNote: This measures the custom paged_attention op only, not full LM generation "
        "(see benchmarks/bench_tps.py for GPT-2 TPS)."
    )


if __name__ == "__main__":
    main()
