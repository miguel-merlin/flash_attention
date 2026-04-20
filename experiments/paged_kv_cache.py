"""
SimplePagedKVCache
==================

An educational, readable paged KV-cache manager for GPT-2-style decode
attention. It intentionally mirrors the memory layout that our custom
``paged_attention`` / ``paged_attention_v2`` kernels expect:

    k_cache[layer]: (P, page_size, H, d)
    v_cache[layer]: (P, page_size, H, d)
    page_table:     (B, max_pages)   int64
    seq_lens:       (B,)             int64

For simplicity (and because this is a teaching prototype, *not* vLLM) we use a
static identity allocation: batch row ``b`` always owns the ``max_pages`` pages
[b*max_pages, (b+1)*max_pages), and logical page ``lp`` maps to physical page
``b * max_pages + lp``.

This file has NO dependency on the compiled CUDA extension, so it runs on CPU
and exposes a CPU self-test under ``if __name__ == "__main__"``.

Usage
-----
    # CPU self-test: round-trips prefill, exercises multi-layer decode at a
    # shared ``position=``, and verifies alternate input shapes. Prints
    # ``SimplePagedKVCache CPU self-test: PASS`` on success.
    python3 experiments/paged_kv_cache.py

Programmatic usage
------------------
    from experiments.paged_kv_cache import SimplePagedKVCache

    cache = SimplePagedKVCache(num_layers=12, num_heads=12, head_dim=64,
                               max_seq_len=1024, page_size=16, batch_size=1)
    cache.write_prefill(layer_idx=0, k=k_prompt, v=v_prompt)   # (B,H,N,d)
    cache.append_decode(layer_idx=0, k_new=k_new, v_new=v_new, position=N)
    k, v, page_table, seq_lens = cache.get_layer_cache(0)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


DeviceLike = Union[torch.device, str]


# ---------------------------------------------------------------------------
# Shape normalization helpers
# ---------------------------------------------------------------------------

def _normalize_prefill_kv(
    t: Tensor,
    *,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    name: str,
) -> Tensor:
    """
    Normalize a prefill K or V tensor to shape ``(B, H, N, d)``.

    Accepts any of:
        (B, H, N, d)
        (B, N, H, d)
        (H, N, d)     -> interpreted as B=1
        (N, H, d)     -> interpreted as B=1

    Raises a clear error for anything else.
    """
    H, d = num_heads, head_dim

    if t.dim() == 4:
        B0, a, b, c = t.shape
        # (B, H, N, d)
        if a == H and c == d:
            return t
        # (B, N, H, d) -> permute to (B, H, N, d)
        if b == H and c == d:
            return t.permute(0, 2, 1, 3).contiguous()
        raise ValueError(
            f"{name} has 4-D shape {tuple(t.shape)}; expected (B,H,N,d) or "
            f"(B,N,H,d) with H={H}, d={d}."
        )

    if t.dim() == 3:
        a, b, c = t.shape
        # (H, N, d) -> (1, H, N, d)
        if a == H and c == d:
            return t.unsqueeze(0)
        # (N, H, d) -> (1, H, N, d)
        if b == H and c == d:
            return t.permute(1, 0, 2).contiguous().unsqueeze(0)
        raise ValueError(
            f"{name} has 3-D shape {tuple(t.shape)}; expected (H,N,d) or "
            f"(N,H,d) with H={H}, d={d}."
        )

    raise ValueError(
        f"{name} must be 3-D or 4-D for prefill; got shape {tuple(t.shape)}."
    )


def _normalize_decode_kv(
    t: Tensor,
    *,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    name: str,
) -> Tensor:
    """
    Normalize a single-token decode K or V tensor to shape ``(B, H, d)``.

    Accepts any of:
        (B, H, d)
        (B, H, 1, d)    -> squeeze the singleton seq dim
        (H, d)          -> interpreted as B=1
    """
    H, d = num_heads, head_dim

    if t.dim() == 4:
        B0, Ht, Nt, dt = t.shape
        if Ht == H and Nt == 1 and dt == d:
            return t.squeeze(2).contiguous()
        raise ValueError(
            f"{name} has 4-D shape {tuple(t.shape)}; expected (B,H,1,d) with "
            f"H={H}, d={d}."
        )

    if t.dim() == 3:
        a, Ht, dt = t.shape
        if Ht == H and dt == d:
            return t
        raise ValueError(
            f"{name} has 3-D shape {tuple(t.shape)}; expected (B,H,d) with "
            f"H={H}, d={d}."
        )

    if t.dim() == 2:
        Ht, dt = t.shape
        if Ht == H and dt == d:
            return t.unsqueeze(0)
        raise ValueError(
            f"{name} has 2-D shape {tuple(t.shape)}; expected (H,d) with "
            f"H={H}, d={d}."
        )

    raise ValueError(
        f"{name} must be 2-D, 3-D or 4-D for decode; got shape {tuple(t.shape)}."
    )


# ---------------------------------------------------------------------------
# SimplePagedKVCache
# ---------------------------------------------------------------------------

class SimplePagedKVCache:
    """
    Minimal paged KV cache compatible with ``paged_attention_v2_forward``.

    Memory layout per layer:
        k_cache[layer]: (P, page_size, H, d)
        v_cache[layer]: (P, page_size, H, d)

    where ``P = batch_size * max_pages`` and
    ``max_pages = ceil(max_seq_len / page_size)``.

    The ``page_table`` is pre-filled with the identity / static assignment
    ``page_table[b, lp] = b * max_pages + lp``. This keeps things educational
    and simple; a real paged allocator would manage free/used pages.

    All tensors live on ``device`` and use ``dtype``. The class is pure-PyTorch
    tensor indexing: it works on CPU as well as CUDA with no custom extension.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        page_size: int = 16,
        batch_size: int = 1,
        device: DeviceLike = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
        if page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {page_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.page_size = int(page_size)
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.max_pages = math.ceil(self.max_seq_len / self.page_size)
        self.total_pages = self.batch_size * self.max_pages

        # Per-layer K/V caches.
        cache_shape = (self.total_pages, self.page_size, self.num_heads, self.head_dim)
        self.k_cache = [
            torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.v_cache = [
            torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]

        # Identity/static page table: page_table[b, lp] = b * max_pages + lp
        row = torch.arange(self.max_pages, dtype=torch.long, device=self.device)
        batch_offset = (
            torch.arange(self.batch_size, dtype=torch.long, device=self.device)
            * self.max_pages
        ).unsqueeze(1)
        self.page_table = (batch_offset + row.unsqueeze(0)).contiguous()  # (B, max_pages)

        self.seq_lens = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Lifecycle / basic API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero the caches and set all ``seq_lens`` back to 0."""
        for layer in range(self.num_layers):
            self.k_cache[layer].zero_()
            self.v_cache[layer].zero_()
        self.seq_lens.zero_()

    def get_layer_cache(
        self, layer_idx: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return ``(k_cache_layer, v_cache_layer, page_table, seq_lens)``.

        Shapes:
            k_cache_layer: (P, page_size, H, d)
            v_cache_layer: (P, page_size, H, d)
            page_table:    (B, max_pages)   int64
            seq_lens:      (B,)             int64
        """
        self._check_layer(layer_idx)
        return (
            self.k_cache[layer_idx],
            self.v_cache[layer_idx],
            self.page_table,
            self.seq_lens,
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write_prefill(
        self,
        layer_idx: int,
        k: Tensor,
        v: Tensor,
        batch_idx: int = 0,
    ) -> None:
        """
        Write the full prompt K/V for one layer / one batch row into the cache.

        Accepts inputs in ``(B,H,N,d)``, ``(B,N,H,d)``, ``(H,N,d)`` or
        ``(N,H,d)`` form; they are normalized internally to ``(B,H,N,d)``.

        Sets ``seq_lens[batch_idx] = N``.
        """
        self._check_layer(layer_idx)
        self._check_batch(batch_idx)

        k_bhnd = _normalize_prefill_kv(
            k,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            name="k",
        )
        v_bhnd = _normalize_prefill_kv(
            v,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            name="v",
        )
        if k_bhnd.shape != v_bhnd.shape:
            raise ValueError(
                f"k and v must have matching shapes after normalization; got "
                f"{tuple(k_bhnd.shape)} vs {tuple(v_bhnd.shape)}."
            )

        B, H, N, d = k_bhnd.shape
        if B != self.batch_size:
            raise ValueError(
                f"write_prefill expected B={self.batch_size}, got B={B}."
            )
        if H != self.num_heads or d != self.head_dim:
            raise ValueError(
                f"write_prefill got (H={H}, d={d}); cache was built with "
                f"(H={self.num_heads}, d={self.head_dim})."
            )
        if N > self.max_seq_len:
            raise ValueError(
                f"write_prefill got N={N} > max_seq_len={self.max_seq_len}."
            )

        k_layer = self.k_cache[layer_idx]
        v_layer = self.v_cache[layer_idx]
        page_table = self.page_table

        b = batch_idx
        # Cast only if necessary (keeps numerical parity tight).
        k_src = k_bhnd[b].to(dtype=self.dtype, device=self.device, copy=False)
        v_src = v_bhnd[b].to(dtype=self.dtype, device=self.device, copy=False)

        for t in range(N):
            logical_page = t // self.page_size
            offset = t % self.page_size
            physical_page = int(page_table[b, logical_page].item())
            # k_src[:, t, :] is (H, d); cache slot is (H, d)
            k_layer[physical_page, offset] = k_src[:, t, :]
            v_layer[physical_page, offset] = v_src[:, t, :]

        self.seq_lens[b] = N

    def append_decode(
        self,
        layer_idx: int,
        k_new: Tensor,
        v_new: Tensor,
        batch_idx: int = 0,
        position: Optional[int] = None,
    ) -> None:
        """
        Append one decode token's K/V for one layer / one batch row.

        Accepts inputs in ``(B,H,d)``, ``(B,H,1,d)`` or ``(H,d)`` form;
        they are normalized internally to ``(B,H,d)``.

        By default (``position=None``) the write position is
        ``seq_lens[batch_idx]`` and ``seq_lens`` is incremented by 1 — useful
        for single-layer tests / cross-check harnesses.

        When writing K/V for the same decode step across many layers, pass
        an explicit ``position`` (the token index this decode step occupies).
        All layers in the step write at the same ``position`` and
        ``seq_lens`` is only advanced via ``max(seq_lens, position + 1)``,
        so seq_lens stays correct regardless of how many layers share it.

        Raises if the cache is already full.
        """
        self._check_layer(layer_idx)
        self._check_batch(batch_idx)

        k_bhd = _normalize_decode_kv(
            k_new,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            name="k_new",
        )
        v_bhd = _normalize_decode_kv(
            v_new,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            name="v_new",
        )
        if k_bhd.shape != v_bhd.shape:
            raise ValueError(
                f"k_new and v_new must have matching shapes after normalization; "
                f"got {tuple(k_bhd.shape)} vs {tuple(v_bhd.shape)}."
            )

        B, H, d = k_bhd.shape
        if B != self.batch_size:
            raise ValueError(
                f"append_decode expected B={self.batch_size}, got B={B}."
            )
        if H != self.num_heads or d != self.head_dim:
            raise ValueError(
                f"append_decode got (H={H}, d={d}); cache was built with "
                f"(H={self.num_heads}, d={self.head_dim})."
            )

        b = batch_idx
        current_len = int(self.seq_lens[b].item())
        if position is None:
            t = current_len
        else:
            t = int(position)
            if t < 0:
                raise ValueError(f"position must be >= 0; got {t}.")
        if t >= self.max_seq_len:
            raise RuntimeError(
                f"append_decode: position {t} >= max_seq_len {self.max_seq_len} "
                f"(batch {b})."
            )

        logical_page = t // self.page_size
        offset = t % self.page_size
        physical_page = int(self.page_table[b, logical_page].item())

        k_layer = self.k_cache[layer_idx]
        v_layer = self.v_cache[layer_idx]

        k_src = k_bhd[b].to(dtype=self.dtype, device=self.device, copy=False)
        v_src = v_bhd[b].to(dtype=self.dtype, device=self.device, copy=False)

        k_layer[physical_page, offset] = k_src  # (H, d)
        v_layer[physical_page, offset] = v_src

        # Advance seq_lens only if this write extended the frontier. This lets
        # multi-layer callers pass the same ``position`` for every layer
        # without double-counting.
        if t + 1 > current_len:
            self.seq_lens[b] = t + 1

    # ------------------------------------------------------------------
    # Reads / reconstruction
    # ------------------------------------------------------------------

    def reconstruct(self, layer_idx: int, batch_idx: int = 0) -> Tensor:
        """
        Reconstruct contiguous K/V for ``batch_idx`` up to the current
        ``seq_lens[batch_idx]``.

        Returns a tuple ``(k, v)`` with shape ``(1, H, N, d)`` each.
        Using a leading batch dim of 1 (not dropping it) keeps the output
        directly comparable with typical ``(B,H,N,d)`` conventions used by
        the rest of the codebase.
        """
        self._check_layer(layer_idx)
        self._check_batch(batch_idx)

        b = batch_idx
        N = int(self.seq_lens[b].item())
        H, d = self.num_heads, self.head_dim

        k_out = torch.zeros((1, H, N, d), dtype=self.dtype, device=self.device)
        v_out = torch.zeros((1, H, N, d), dtype=self.dtype, device=self.device)

        if N == 0:
            return k_out, v_out

        k_layer = self.k_cache[layer_idx]
        v_layer = self.v_cache[layer_idx]

        for t in range(N):
            logical_page = t // self.page_size
            offset = t % self.page_size
            physical_page = int(self.page_table[b, logical_page].item())
            k_out[0, :, t, :] = k_layer[physical_page, offset]
            v_out[0, :, t, :] = v_layer[physical_page, offset]

        return k_out, v_out

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_layer(self, layer_idx: int) -> None:
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(
                f"layer_idx={layer_idx} out of range [0, {self.num_layers})."
            )

    def _check_batch(self, batch_idx: int) -> None:
        if not (0 <= batch_idx < self.batch_size):
            raise IndexError(
                f"batch_idx={batch_idx} out of range [0, {self.batch_size})."
            )

    def __repr__(self) -> str:
        return (
            f"SimplePagedKVCache(num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"max_seq_len={self.max_seq_len}, page_size={self.page_size}, "
            f"batch_size={self.batch_size}, total_pages={self.total_pages}, "
            f"device={self.device}, dtype={self.dtype})"
        )


# ---------------------------------------------------------------------------
# CPU self-test
# ---------------------------------------------------------------------------

def _cpu_self_test() -> None:
    torch.manual_seed(0)

    num_layers = 2
    num_heads = 4
    head_dim = 8
    max_seq_len = 64
    page_size = 16
    batch_size = 1
    device = torch.device("cpu")
    dtype = torch.float32

    cache = SimplePagedKVCache(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        page_size=page_size,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    print(cache)

    # Prompt of length N (deliberately non-multiple of page_size to exercise pages)
    N = 20
    k_prompt = torch.randn(batch_size, num_heads, N, head_dim, dtype=dtype)
    v_prompt = torch.randn(batch_size, num_heads, N, head_dim, dtype=dtype)

    # Write into layer 0 and layer 1 with different tensors to make sure
    # per-layer isolation works.
    cache.write_prefill(0, k_prompt, v_prompt)
    k_prompt2 = torch.randn_like(k_prompt)
    v_prompt2 = torch.randn_like(v_prompt)
    cache.write_prefill(1, k_prompt2, v_prompt2)

    assert int(cache.seq_lens[0].item()) == N, "seq_lens after prefill should equal N"

    # Reconstruct layer 0 and compare.
    k0_recon, v0_recon = cache.reconstruct(0)
    assert k0_recon.shape == (1, num_heads, N, head_dim), (
        f"unexpected reconstruct shape: {k0_recon.shape}"
    )
    assert torch.allclose(k0_recon, k_prompt, atol=0, rtol=0), "layer 0 K reconstruction mismatch"
    assert torch.allclose(v0_recon, v_prompt, atol=0, rtol=0), "layer 0 V reconstruction mismatch"

    # Reconstruct layer 1 and compare.
    k1_recon, v1_recon = cache.reconstruct(1)
    assert torch.allclose(k1_recon, k_prompt2, atol=0, rtol=0), "layer 1 K reconstruction mismatch"
    assert torch.allclose(v1_recon, v_prompt2, atol=0, rtol=0), "layer 1 V reconstruction mismatch"

    # Append a decode token for each layer.
    k_new0 = torch.randn(batch_size, num_heads, head_dim, dtype=dtype)
    v_new0 = torch.randn(batch_size, num_heads, head_dim, dtype=dtype)
    k_new1 = torch.randn(batch_size, num_heads, head_dim, dtype=dtype)
    v_new1 = torch.randn(batch_size, num_heads, head_dim, dtype=dtype)

    # Multi-layer decode step: both layers must write at the SAME token
    # position (N) and seq_lens must advance exactly once. This is the
    # expected path in a real LLM where seq_lens is shared across layers.
    cache.append_decode(0, k_new0, v_new0, position=N)
    cache.append_decode(1, k_new1, v_new1, position=N)

    assert int(cache.seq_lens[0].item()) == N + 1, (
        f"seq_lens should advance by 1 per decode step across layers; got "
        f"{int(cache.seq_lens[0].item())}, expected {N + 1}"
    )

    # Reconstruct layer 0 after decode; last row must equal k_new0/v_new0.
    k0_recon2, v0_recon2 = cache.reconstruct(0)
    assert k0_recon2.shape == (1, num_heads, N + 1, head_dim)
    assert torch.allclose(k0_recon2[:, :, :N, :], k_prompt), "prefill part drifted after decode"
    assert torch.allclose(k0_recon2[0, :, N, :], k_new0[0]), "appended K token mismatch"
    assert torch.allclose(v0_recon2[0, :, N, :], v_new0[0]), "appended V token mismatch"

    # And layer 1 must have its own K/V at the same position N — this was
    # the symptom of the multi-layer seq_lens bug fixed by ``position=``.
    k1_recon2, v1_recon2 = cache.reconstruct(1)
    assert k1_recon2.shape == (1, num_heads, N + 1, head_dim)
    assert torch.allclose(k1_recon2[:, :, :N, :], k_prompt2), "layer 1 prefill drifted"
    assert torch.allclose(k1_recon2[0, :, N, :], k_new1[0]), "layer 1 appended K mismatch"
    assert torch.allclose(v1_recon2[0, :, N, :], v_new1[0]), "layer 1 appended V mismatch"

    # Reset and verify seq_lens go back to zero and reconstruct returns empty.
    cache.reset()
    assert int(cache.seq_lens[0].item()) == 0
    k_empty, v_empty = cache.reconstruct(0)
    assert k_empty.shape == (1, num_heads, 0, head_dim)
    assert v_empty.shape == (1, num_heads, 0, head_dim)

    # Exercise alternate input shapes for prefill:
    #   (N, H, d) path
    k_alt = torch.randn(N, num_heads, head_dim, dtype=dtype)
    v_alt = torch.randn(N, num_heads, head_dim, dtype=dtype)
    cache.write_prefill(0, k_alt, v_alt)
    k_alt_recon, v_alt_recon = cache.reconstruct(0)
    # Expected normalized layout is (1, H, N, d)
    expected_k = k_alt.permute(1, 0, 2).unsqueeze(0)
    expected_v = v_alt.permute(1, 0, 2).unsqueeze(0)
    assert torch.allclose(k_alt_recon, expected_k), "(N,H,d) prefill reconstruction mismatch"
    assert torch.allclose(v_alt_recon, expected_v), "(N,H,d) prefill reconstruction mismatch"

    # Exercise alternate input shape for decode: (H, d)
    k_new_alt = torch.randn(num_heads, head_dim, dtype=dtype)
    v_new_alt = torch.randn(num_heads, head_dim, dtype=dtype)
    cache.append_decode(0, k_new_alt, v_new_alt)
    k_alt_recon2, v_alt_recon2 = cache.reconstruct(0)
    assert k_alt_recon2.shape == (1, num_heads, N + 1, head_dim)
    assert torch.allclose(k_alt_recon2[0, :, N, :], k_new_alt), "(H,d) decode append mismatch"
    assert torch.allclose(v_alt_recon2[0, :, N, :], v_new_alt), "(H,d) decode append mismatch"

    print("SimplePagedKVCache CPU self-test: PASS")


if __name__ == "__main__":
    _cpu_self_test()
