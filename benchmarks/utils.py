"""Shared helpers for flash attention benchmarks."""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional


def make_qkv(
    B: int,
    H: int,
    N: int,
    d: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Create reproducible random Q, K, V tensors with shape (B, H, N, d).

    Args:
        B: Batch size
        H: Number of heads
        N: Sequence length
        d: Head dimension
        device: Target device
        dtype:  Tensor dtype
        seed:   RNG seed for reproducibility

    Returns:
        (q, k, v) tuple of tensors on the requested device/dtype
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    shape = (B, H, N, d)
    q = torch.randn(shape, dtype=dtype, generator=gen).to(device)
    k = torch.randn(shape, dtype=dtype, generator=gen).to(device)
    v = torch.randn(shape, dtype=dtype, generator=gen).to(device)
    return q, k, v


def format_bytes(n: int) -> str:
    """Pretty-print a byte count."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} PB"


def format_time_us(seconds: float) -> str:
    """Pretty-print seconds as microseconds or milliseconds."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    return f"{seconds * 1e3:.2f} ms"


def report_table(results: list[dict]) -> None:
    """
    Print a formatted table from a list of result dicts.

    Each dict must have at least a 'config' key; all other keys become columns.

    Example:
        results = [
            {"config": "B=1,H=1,N=64,d=32", "VanillaAttention": "1.2 ms", ...},
        ]
    """
    if not results:
        print("No results.")
        return

    all_keys = list(results[0].keys())
    col_widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in results)) for k in all_keys}
    header = "  ".join(k.ljust(col_widths[k]) for k in all_keys)
    sep = "  ".join("-" * col_widths[k] for k in all_keys)

    print(header)
    print(sep)
    for row in results:
        print("  ".join(str(row.get(k, "")).ljust(col_widths[k]) for k in all_keys))
