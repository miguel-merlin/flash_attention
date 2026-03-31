"""
Attention module variants as nn.Module subclasses.

All modules share the tensor convention:
    q, k, v: (B, H, N, d)
    output:  (B, H, N, d)

where:
  B - batch size
  H - number of heads
  N - sequence length
  d - head dimension
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

try:
    import flash_attn._C
except ImportError:
    pass



class VanillaAttention(nn.Module):
    """
    Standard scaled dot-product attention implemented entirely in PyTorch.

    Works on both CPU and CUDA. Fully differentiable via PyTorch autograd.
    Memory cost is O(N^2) because the full NxN attention matrix is materialised.
    """

    def __init__(self, scale: Optional[float] = None):
        """
        Args:
            scale: Optional explicit scale factor applied to QK^T scores.
                   Defaults to 1/sqrt(d) computed at forward-time.
        """
        super().__init__()
        self._scale = scale

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Args:
            q: Query tensor  (B, H, N, d)
            k: Key tensor    (B, H, N, d)
            v: Value tensor  (B, H, N, d)

        Returns:
            Output tensor    (B, H, N, d)
        """
        if q.dim() != 4:
            raise ValueError(f"Expected 4-D tensors (B, H, N, d), got {q.dim()}-D")
        if q.shape != k.shape or k.shape != v.shape:
            raise ValueError(f"q, k, v must have the same shape. Got {q.shape}, {k.shape}, {v.shape}")

        d = q.shape[-1]
        scale = self._scale if self._scale is not None else 1.0 / math.sqrt(d)

        # (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        probs = torch.softmax(scores, dim=-1)

        # (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
        return torch.matmul(probs, v)

    def extra_repr(self) -> str:
        return f"scale={self._scale}"


# ---------------------------------------------------------------------------
# Custom autograd Function used by FlashAttentionCPP and FlashAttentionCUDA.
# The forward call dispatches to our registered C++ op; the backward receives
# stub gradients now and can be replaced with the real CUDA backward later.
# ---------------------------------------------------------------------------

class _FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor) -> Tensor:  # type: ignore[override]
        # Dispatch to the C++ / CUDA kernel registered under 'flash_attention'
        out = torch.ops.flash_attention.flash_attention(q, k, v)
        ctx.save_for_backward(q, k, v, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        """
        Backward stub — returns analytically correct gradients for vanilla
        attention so the autograd graph is valid.  Replace the body with your
        CUDA backward kernel once it is ready.

        Gradient derivation (standard scaled dot-product attention):
            S   = Q K^T * scale              (B, H, N, N)
            P   = softmax(S)                 (B, H, N, N)
            O   = P V                        (B, H, N, d)

            dV  = P^T dO                     (B, H, N, d)
            dP  = dO V^T                     (B, H, N, N)
            dS  = P * (dP - (dP * P).sum(-1, keepdim=True))   (softmax bwd)
            dQ  = dS K * scale               (B, H, N, d)
            dK  = dS^T Q * scale             (B, H, N, d)
        """
        q, k, v, out = ctx.saved_tensors
        d = q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        # Recompute forward intermediates (no extra memory saved in forward)
        S = torch.matmul(q, k.transpose(-1, -2)) * scale   # (B, H, N, N)
        P = torch.softmax(S, dim=-1)                        # (B, H, N, N)

        # dV
        dV = torch.matmul(P.transpose(-1, -2), grad_output)  # (B, H, N, d)

        # dP
        dP = torch.matmul(grad_output, v.transpose(-1, -2))  # (B, H, N, N)

        # Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
        dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))   # (B, H, N, N)

        # dQ, dK
        dQ = torch.matmul(dS, k) * scale        # (B, H, N, d)
        dK = torch.matmul(dS.transpose(-1, -2), q) * scale  # (B, H, N, d)

        return dQ, dK, dV


class FlashAttentionCPP(nn.Module):
    """
    Flash Attention dispatched through the registered C++ CPU kernel.

    Requires the extension to be compiled:
        pip install -e /home2/mmerlin/flash_attention

    Falls back with a clear error if the extension is not found.
    Gradients are computed via the analytical backward in _FlashAttentionFunction.
    """

    def __init__(self):
        super().__init__()
        self._check_extension()

    @staticmethod
    def _check_extension():
        try:
            torch.ops.flash_attention.flash_attention
        except AttributeError:
            raise RuntimeError(
                "flash_attention C++ extension not found. "
                "Compile it with: pip install -e /home2/mmerlin/flash_attention"
            )

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Args:
            q: Query tensor  (B, H, N, d) — float32, CPU
            k: Key tensor    (B, H, N, d) — float32, CPU
            v: Value tensor  (B, H, N, d) — float32, CPU

        Returns:
            Output tensor    (B, H, N, d)
        """
        if q.device.type != "cpu":
            raise ValueError("FlashAttentionCPP expects CPU tensors. Use FlashAttentionCUDA for GPU.")
        return _FlashAttentionFunction.apply(q, k, v)


class FlashAttentionCUDA(nn.Module):
    """
    Flash Attention dispatched through the registered CUDA kernel.

    Requires both the extension to be compiled AND a CUDA device.
    This is a placeholder; the CUDA kernel body (flash_attention.cu) is a stub
    and should be filled in with the real Flash Attention algorithm.

    Falls back with a clear error if no CUDA device is available.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: Target CUDA device. Defaults to cuda:0.
        """
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("FlashAttentionCUDA requires a CUDA-capable GPU.")
        self.device = device or torch.device("cuda", 0)
        self._check_extension()

    @staticmethod
    def _check_extension():
        try:
            torch.ops.flash_attention.flash_attention
        except AttributeError:
            raise RuntimeError(
                "flash_attention C++ extension not found. "
                "Compile it with: pip install -e /home2/mmerlin/flash_attention"
            )

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Args:
            q: Query tensor  (B, H, N, d) — float32, CUDA
            k: Key tensor    (B, H, N, d) — float32, CUDA
            v: Value tensor  (B, H, N, d) — float32, CUDA

        Returns:
            Output tensor    (B, H, N, d)
        """
        if q.device.type != "cuda":
            q = q.to(self.device)
            k = k.to(self.device)
            v = v.to(self.device)
        return _FlashAttentionFunction.apply(q, k, v)


# ---------------------------------------------------------------------------
# Native Vanilla Attention Wrappers
# ---------------------------------------------------------------------------

class _NativeVanillaAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor) -> Tensor:  # type: ignore[override]
        out = torch.ops.vanilla_attention.vanilla_attention(q, k, v)
        ctx.save_for_backward(q, k, v, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        q, k, v, out = ctx.saved_tensors
        d = q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        
        # Softmax derivatives in python
        S = torch.matmul(q, k.transpose(-1, -2)) * scale
        P = torch.softmax(S, dim=-1)
        
        dV = torch.matmul(P.transpose(-1, -2), grad_output)
        dP = torch.matmul(grad_output, v.transpose(-1, -2))
        dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
        
        dQ = torch.matmul(dS, k) * scale
        dK = torch.matmul(dS.transpose(-1, -2), q) * scale
        
        return dQ, dK, dV

class NativeVanillaAttentionCPP(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            torch.ops.vanilla_attention.vanilla_attention
        except AttributeError:
            raise RuntimeError("vanilla_attention C++ extension not found.")

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if q.device.type != "cpu":
            raise ValueError("NativeVanillaAttentionCPP expects CPU tensors.")
        return _NativeVanillaAttentionFunction.apply(q, k, v)


class NativeVanillaAttentionCUDA(nn.Module):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("NativeVanillaAttentionCUDA requires a CUDA GPU.")
        self.device = device or torch.device("cuda", 0)
        try:
            torch.ops.vanilla_attention.vanilla_attention
        except AttributeError:
            raise RuntimeError("vanilla_attention C++ extension not found.")

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if q.device.type != "cuda":
            q = q.to(self.device)
            k = k.to(self.device)
            v = v.to(self.device)
        return _NativeVanillaAttentionFunction.apply(q, k, v)
