"""Shared helpers for flash attention benchmarks."""

from __future__ import annotations

import torch
from torch import Tensor


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


def make_paged_attention_inputs(
    B: int,
    H: int,
    N: int,
    d: int,
    *,
    page_size: int = 16,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Build tensors for ``flash_attention::paged_attention`` (decode over ``N`` cached tokens).

    ``q`` is (B, H, d); ``k_cache`` / ``v_cache`` are (P, page_size, H, d) with identity
    ``page_table`` so position ``t`` maps to physical page ``t // page_size``;
    ``seq_lens[b] == N`` for all ``b``.

    Returns:
        (q, k_cache, v_cache, page_table, seq_lens)
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    max_pages = max(1, (N + page_size - 1) // page_size)
    P = max_pages
    q = torch.randn(B, H, d, dtype=dtype, generator=gen).to(device)
    shape_cache = (P, page_size, H, d)
    k_cache = torch.randn(shape_cache, dtype=dtype, generator=gen).to(device)
    v_cache = torch.randn(shape_cache, dtype=dtype, generator=gen).to(device)
    page_table = (
        torch.arange(max_pages, dtype=torch.long)
        .unsqueeze(0)
        .expand(B, -1)
        .contiguous()
        .to(device)
    )
    seq_lens = torch.full((B,), N, dtype=torch.long, device=device)
    return q, k_cache, v_cache, page_table, seq_lens


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
