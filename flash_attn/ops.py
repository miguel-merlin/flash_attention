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

__all__ = ["flash_attention_forward", "register_ops"]

# Track whether we have already registered the fake + autograd hooks
_OPS_REGISTERED = False


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
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    S = torch.matmul(q, k.transpose(-1, -2)) * scale
    P = torch.softmax(S, dim=-1)

    dV = torch.matmul(P.transpose(-1, -2), grad_output)
    dP = torch.matmul(grad_output, v.transpose(-1, -2))
    dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
    dQ = torch.matmul(dS, k) * scale
    dK = torch.matmul(dS.transpose(-1, -2), q) * scale

    return dQ, dK, dV


def _setup_context(ctx, inputs, outputs):
    q, k, v = inputs
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
    def _flash_attention_fake(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        torch._check(q.shape == k.shape == v.shape,
                     lambda: "q, k, v must have the same shape")
        torch._check(q.dtype == k.dtype == v.dtype,
                     lambda: "q, k, v must have the same dtype")
        torch._check(q.device == k.device == v.device,
                     lambda: "q, k, v must be on the same device")
        return torch.empty_like(q)

    # Autograd wiring
    torch.library.register_autograd(
        "flash_attention::flash_attention",
        _backward,
        setup_context=_setup_context,
    )

    _OPS_REGISTERED = True


def flash_attention_forward(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """
    Dispatch flash attention through the registered C++ / CUDA op.

    Args:
        q: (B, H, N, d)  — same device/dtype as k and v
        k: (B, H, N, d)
        v: (B, H, N, d)

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
    return op(q, k, v)