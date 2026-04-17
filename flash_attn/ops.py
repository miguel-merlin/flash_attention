"""
Low-level op registration for the flash_attention C++ extension.

This module:
  1. Registers a fake/meta implementation for torch.compile / export support
  2. Wires autograd (backward) for the registered op
  3. Exposes a clean Python entry-point: flash_attention_forward(q, k, v)

IMPORTANT: The fake impl and autograd registration are deferred until the
C++ extension is actually loaded (i.e. flash_attention._EXTENSION_LOADED is
True). Without the compiled extension, only flash_attention_forward (which
raises a clear error) and the module itself are importable.
"""

import math
import torch
from torch import Tensor

__all__ = [
    "flash_attention_forward",
    "register_ops",
    "paged_attention_forward",
    "register_paged_ops",
]

# Track whether we have already registered the fake + autograd hooks
_OPS_REGISTERED = False
_PAGED_OPS_REGISTERED = False


def _backward(ctx, grad_output: Tensor):
    """
    Analytically correct gradients for scaled dot-product attention.
    Replace this body with your custom CUDA backward kernel once it is ready.

        S = Q K^T * scale
        P = softmax(S)
        O = P V

        dV = P^T dO
        dP = dO V^T
        dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))   [softmax bwd]
        dQ = dS K * scale
        dK = dS^T Q * scale
    """
    q, k, v = ctx.saved_tensors
    is_causal = getattr(ctx, "is_causal", False)
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    S = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(S.shape[-2:], dtype=torch.bool, device=S.device), diagonal=1
        )
        S = S.masked_fill(causal_mask, float("-inf"))
    P = torch.softmax(S, dim=-1)

    dV = torch.matmul(P.transpose(-1, -2), grad_output)
    dP = torch.matmul(grad_output, v.transpose(-1, -2))
    dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
    dQ = torch.matmul(dS, k) * scale
    dK = torch.matmul(dS.transpose(-1, -2), q) * scale

    return dQ, dK, dV, None


def _setup_context(ctx, inputs, outputs):
    q, k, v = inputs[0], inputs[1], inputs[2]
    ctx.is_causal = bool(inputs[3]) if len(inputs) > 3 else False
    saved_q = q if ctx.needs_input_grad[0] else None
    saved_k = k if ctx.needs_input_grad[1] else None
    saved_v = v if ctx.needs_input_grad[2] else None
    ctx.save_for_backward(saved_q, saved_k, saved_v)


def register_ops() -> None:
    """
    Register the fake impl and autograd hook for flash_attention::flash_attention.

    Must be called AFTER the C++ extension has been imported (which triggers
    TORCH_LIBRARY registration), otherwise the op does not yet exist and
    torch.library will raise RuntimeError("operator … does not exist").

    This is called automatically from flash_attn/__init__.py when _C loads.
    """
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return

    # Fake / abstract implementation (for torch.compile, torch.export, etc.)
    @torch.library.register_fake("flash_attention::flash_attention")
    def _flash_attention_fake(q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
        torch._check(q.shape == k.shape == v.shape,
                     lambda: "q, k, v must have the same shape")
        torch._check(q.dtype == k.dtype == v.dtype,
                     lambda: "q, k, v must have the same dtype")
        torch._check(q.device == k.device == v.device,
                     lambda: "q, k, v must be on the same device")
        _ = is_causal  # meta-only; matches C++ schema
        return torch.empty_like(q)

    # Autograd wiring
    torch.library.register_autograd(
        "flash_attention::flash_attention",
        _backward,
        setup_context=_setup_context,
    )

    _OPS_REGISTERED = True


def flash_attention_forward(
    q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False
) -> Tensor:
    """
    Dispatch flash attention through the registered C++ / CUDA op.

    Args:
        q: (B, H, N, d)  — same device/dtype as k and v
        k: (B, H, N, d)
        v: (B, H, N, d)
        is_causal: if True, apply causal masking in the kernel / reference path.

    Returns:
        (B, H, N, d) output tensor

    Raises:
        RuntimeError: if the C++ extension has not been compiled.
    """
    try:
        op = torch.ops.flash_attention.flash_attention
    except AttributeError:
        raise RuntimeError(
            "flash_attention C++ extension not loaded. "
            "Compile it with: pip install -e /home2/mmerlin/flash_attention"
        )
    return op(q, k, v, is_causal)


def register_paged_ops() -> None:
    """
    Register the fake/meta implementation for flash_attention::paged_attention.

    Call after the C++ extension is imported. CUDA/CPU dispatch is handled in C++;
    C++ backward stubs are not implemented — use a Python autograd.Function if you
    need gradients.
    """
    global _PAGED_OPS_REGISTERED
    if _PAGED_OPS_REGISTERED:
        return

    @torch.library.register_fake("flash_attention::paged_attention")
    def _paged_attention_fake(
        q: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        page_table: Tensor,
        seq_lens: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        torch._check(q.dim() == 3, lambda: "q must be (B, H, d)")
        torch._check(k_cache.dim() == 4, lambda: "k_cache must be (P, page_size, H, d)")
        torch._check(v_cache.shape == k_cache.shape, lambda: "v_cache must match k_cache")
        torch._check(page_table.dim() == 2, lambda: "page_table must be (B, max_pages)")
        torch._check(seq_lens.dim() == 1, lambda: "seq_lens must be (B,)")
        torch._check(
            q.shape[0] == page_table.shape[0] == seq_lens.shape[0],
            lambda: "B mismatch among q, page_table, seq_lens",
        )
        _ = is_causal
        B, H, d = q.shape
        return torch.empty((B, H, d), dtype=q.dtype, device=q.device)

    _PAGED_OPS_REGISTERED = True


def paged_attention_forward(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    page_table: Tensor,
    seq_lens: Tensor,
    is_causal: bool = False,
) -> Tensor:
    """
    Paged attention for a single decoded token per batch: q is (B, H, d); keys/values
    live in (P, page_size, H, d) caches addressed by ``page_table`` (B, max_pages).

    ``is_causal`` matches the flash_attention / vanilla_attention API; for the
    current decode-only layout (attention over positions ``0 .. seq_len-1``) it
    does not change numerics.
    """
    try:
        op = torch.ops.flash_attention.paged_attention
    except AttributeError:
        raise RuntimeError(
            "flash_attention C++ extension not loaded. "
            "Compile with: pip install -e . --no-build-isolation"
        ) from None
    return op(q, k_cache, v_cache, page_table, seq_lens, is_causal)