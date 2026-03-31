"""
Correctness tests for all attention module variants.

Usage:
    source .venv/bin/activate
    python3 -m pytest tests/test_attention.py -v
  or
    python3 tests/test_attention.py

Tests:
  - VanillaAttention: shape, dtype, matches reference implementation
  - FlashAttentionCPP: matches VanillaAttention (skipped if extension not built)
  - Backward: gradient check via torch.autograd.gradcheck for VanillaAttention
"""

from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import Tensor

from flash_attn.attention import VanillaAttention
from flash_attn.reference import attention_reference_torch

ATOL = 1e-5
RTOL = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkv(B, H, N, d, *, dtype=torch.float32, device="cpu") -> tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(0)
    return (
        torch.randn(B, H, N, d, dtype=dtype, device=device),
        torch.randn(B, H, N, d, dtype=dtype, device=device),
        torch.randn(B, H, N, d, dtype=dtype, device=device),
    )


# ---------------------------------------------------------------------------
# VanillaAttention tests
# ---------------------------------------------------------------------------

class TestVanillaAttention(unittest.TestCase):
    def setUp(self):
        self.attn = VanillaAttention()

    def test_output_shape(self):
        for B, H, N, d in [(1, 1, 4, 8), (2, 4, 16, 32)]:
            with self.subTest(B=B, H=H, N=N, d=d):
                q, k, v = make_qkv(B, H, N, d)
                out = self.attn(q, k, v)
                self.assertEqual(out.shape, (B, H, N, d))

    def test_matches_reference(self):
        """VanillaAttention must agree with the manual reference to tight tolerance."""
        for B, H, N, d in [(1, 1, 4, 4), (2, 2, 8, 8)]:
            with self.subTest(B=B, H=H, N=N, d=d):
                q, k, v = make_qkv(B, H, N, d)
                out_vanilla = self.attn(q, k, v)
                out_ref = attention_reference_torch(q, k, v)
                self.assertTrue(
                    torch.allclose(out_vanilla, out_ref, atol=ATOL, rtol=RTOL),
                    msg=f"Max diff: {(out_vanilla - out_ref).abs().max().item():.2e}",
                )

    def test_wrong_ndim_raises(self):
        with self.assertRaises(ValueError):
            q = torch.randn(4, 8)   # 2-D instead of 4-D
            self.attn(q, q, q)

    def test_shape_mismatch_raises(self):
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 16)  # wrong d
        v = torch.randn(1, 1, 4, 8)
        with self.assertRaises(ValueError):
            self.attn(q, k, v)

    def test_output_dtype_preserved(self):
        q, k, v = make_qkv(1, 1, 4, 4)
        out = self.attn(q, k, v)
        self.assertEqual(out.dtype, torch.float32)

    def test_probabilities_sum_to_one(self):
        """Attention probabilities (P = softmax(QK^T/sqrt(d))) must sum to 1 per row."""
        B, H, N, d = 1, 1, 8, 4
        q, k, v = make_qkv(B, H, N, d)
        import math
        scale = 1.0 / math.sqrt(d)
        S = torch.matmul(q, k.transpose(-1, -2)) * scale
        P = torch.softmax(S, dim=-1)
        row_sums = P.sum(dim=-1)
        self.assertTrue(
            torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6),
            msg="Softmax rows do not sum to 1",
        )


# ---------------------------------------------------------------------------
# FlashAttentionCPP tests (skipped when extension not compiled)
# ---------------------------------------------------------------------------

_SKIP_CPP = False
_SKIP_CPP_REASON = ""
try:
    from flash_attn.attention import FlashAttentionCPP
    _flash_cpp = FlashAttentionCPP()
except RuntimeError as _e:
    _SKIP_CPP = True
    _SKIP_CPP_REASON = str(_e)


@unittest.skipIf(_SKIP_CPP, f"FlashAttentionCPP not available: {_SKIP_CPP_REASON}")
class TestFlashAttentionCPP(unittest.TestCase):
    def setUp(self):
        self.flash = _flash_cpp
        self.vanilla = VanillaAttention()

    def test_output_shape(self):
        for B, H, N, d in [(1, 1, 4, 4), (2, 2, 8, 8)]:
            with self.subTest(B=B, H=H, N=N, d=d):
                q, k, v = make_qkv(B, H, N, d)
                out = self.flash(q, k, v)
                self.assertEqual(out.shape, (B, H, N, d))

    def test_matches_vanilla(self):
        """FlashAttentionCPP output must match VanillaAttention within tolerance."""
        for B, H, N, d in [(1, 1, 4, 4), (2, 2, 8, 8), (1, 4, 16, 8)]:
            with self.subTest(B=B, H=H, N=N, d=d):
                q, k, v = make_qkv(B, H, N, d)
                out_flash = self.flash(q, k, v)
                out_vanilla = self.vanilla(q, k, v)
                self.assertTrue(
                    torch.allclose(out_flash, out_vanilla, atol=ATOL, rtol=RTOL),
                    msg=f"Max diff: {(out_flash - out_vanilla).abs().max().item():.2e}",
                )


# ---------------------------------------------------------------------------
# Backward / autograd tests
# ---------------------------------------------------------------------------

class TestBackward(unittest.TestCase):
    """Gradient checks for the Python-level autograd."""

    def test_vanilla_gradcheck(self):
        """
        Verify that VanillaAttention has correct gradients via gradcheck.
        Uses double precision and small tensors (gradcheck is slow).
        """
        B, H, N, d = 1, 1, 4, 4
        torch.manual_seed(1)
        q = torch.randn(B, H, N, d, dtype=torch.float64, requires_grad=True)
        k = torch.randn(B, H, N, d, dtype=torch.float64, requires_grad=True)
        v = torch.randn(B, H, N, d, dtype=torch.float64, requires_grad=True)

        vanilla = VanillaAttention()

        try:
            torch.autograd.gradcheck(
                vanilla,
                (q, k, v),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
            )
            passed = True
        except Exception as e:
            passed = False
            self.fail(f"gradcheck failed for VanillaAttention: {e}")

    def test_vanilla_backward_shapes(self):
        """Gradients must have the same shape as the inputs."""
        B, H, N, d = 2, 2, 8, 4
        q, k, v = make_qkv(B, H, N, d)
        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
        v = v.requires_grad_(True)

        vanilla = VanillaAttention()
        out = vanilla(q, k, v)
        out.sum().backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)
        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(k.grad.shape, k.shape)
        self.assertEqual(v.grad.shape, v.shape)

    def test_vanilla_backward_values(self):
        """
        Simple numerical gradient check: finite-difference vs autograd.
        Tolerance is relaxed compared to gradcheck because we use float32.
        """
        B, H, N, d = 1, 1, 4, 4
        eps = 1e-3
        atol = 1e-2

        torch.manual_seed(2)
        q0 = torch.randn(B, H, N, d)
        k0 = torch.randn(B, H, N, d)
        v0 = torch.randn(B, H, N, d)

        q = q0.clone().requires_grad_(True)
        k = k0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)

        vanilla = VanillaAttention()
        out = vanilla(q, k, v)
        grad_out = torch.ones_like(out)
        out.backward(grad_out)

        dq_auto = q.grad.clone()

        # Finite-difference estimate for dq over first element
        q_plus = q0.clone()
        q_plus[0, 0, 0, 0] += eps
        q_minus = q0.clone()
        q_minus[0, 0, 0, 0] -= eps

        f_plus  = (vanilla(q_plus, k0, v0) * grad_out).sum()
        f_minus = (vanilla(q_minus, k0, v0) * grad_out).sum()
        dq_fd = (f_plus - f_minus) / (2 * eps)

        self.assertAlmostEqual(
            dq_auto[0, 0, 0, 0].item(),
            dq_fd.item(),
            delta=atol,
            msg=f"Autograd dq={dq_auto[0,0,0,0]:.6f}  FD dq={dq_fd:.6f}",
        )


# ---------------------------------------------------------------------------
# FlashAttentionCUDA tests (skipped when extension not compiled or no CUDA)
# ---------------------------------------------------------------------------

_SKIP_CUDA = False
_SKIP_CUDA_REASON = ""
try:
    from flash_attn.attention import FlashAttentionCUDA
    _flash_cuda = FlashAttentionCUDA()
except RuntimeError as _e:
    _SKIP_CUDA = True
    _SKIP_CUDA_REASON = str(_e)


@unittest.skipIf(_SKIP_CUDA, f"FlashAttentionCUDA not available: {_SKIP_CUDA_REASON}")
class TestFlashAttentionCUDA(unittest.TestCase):
    def setUp(self):
        self.flash = _flash_cuda
        self.vanilla = VanillaAttention()

    def test_output_shape(self):
        for B, H, N, d in [(1, 1, 4, 4), (2, 2, 8, 8)]:
            with self.subTest(B=B, H=H, N=N, d=d):
                q, k, v = make_qkv(B, H, N, d, device="cuda")
                out = self.flash(q, k, v)
                self.assertEqual(out.shape, (B, H, N, d))

    def test_matches_vanilla(self):
        """FlashAttentionCUDA output must match VanillaAttention within tolerance."""
        for B, H, N, d in [(1, 1, 4, 4), (2, 2, 32, 32), (1, 4, 128, 64)]:
            with self.subTest(B=B, H=H, N=N, d=d):
                # FlashAttentionCUDA wrapper automatically moves to CUDA if needed.
                q, k, v = make_qkv(B, H, N, d, device="cpu")
                
                out_flash = self.flash(q, k, v).cpu()
                out_vanilla = self.vanilla(q, k, v)
                self.assertTrue(
                    torch.allclose(out_flash, out_vanilla, atol=ATOL, rtol=RTOL),
                    msg=f"Max diff: {(out_flash - out_vanilla).abs().max().item():.2e}",
                )

# ---------------------------------------------------------------------------
# Native Vanilla Attention (CPP and CUDA) Tests
# ---------------------------------------------------------------------------

_SKIP_NATIVE_CPP = False
_SKIP_NATIVE_CPP_REASON = ""
try:
    from flash_attn.attention import NativeVanillaAttentionCPP
    _native_cpp = NativeVanillaAttentionCPP()
except RuntimeError as _e:
    _SKIP_NATIVE_CPP = True
    _SKIP_NATIVE_CPP_REASON = str(_e)

@unittest.skipIf(_SKIP_NATIVE_CPP, f"NativeVanillaAttentionCPP not available: {_SKIP_NATIVE_CPP_REASON}")
class TestNativeVanillaAttentionCPP(unittest.TestCase):
    def setUp(self):
        self.attn = _native_cpp
        self.vanilla = VanillaAttention()

    def test_output_shape(self):
        for B, H, N, d in [(1, 1, 4, 4), (2, 2, 8, 8)]:
            with self.subTest(B=B, H=H, N=N, d=d):
                q, k, v = make_qkv(B, H, N, d, device="cpu")
                out = self.attn(q, k, v)
                self.assertEqual(out.shape, (B, H, N, d))

    def test_matches_vanilla(self):
        for B, H, N, d in [(1, 1, 4, 4), (2, 2, 8, 8), (1, 4, 16, 8)]:
            with self.subTest(B=B, H=H, N=N, d=d):
                q, k, v = make_qkv(B, H, N, d, device="cpu")
                out_native = self.attn(q, k, v)
                out_vanilla = self.vanilla(q, k, v)
                self.assertTrue(
                    torch.allclose(out_native, out_vanilla, atol=ATOL, rtol=RTOL),
                    msg=f"Max diff: {(out_native - out_vanilla).abs().max().item():.2e}",
                )

_SKIP_NATIVE_CUDA = False
_SKIP_NATIVE_CUDA_REASON = ""
try:
    from flash_attn.attention import NativeVanillaAttentionCUDA
    _native_cuda = NativeVanillaAttentionCUDA()
except RuntimeError as _e:
    _SKIP_NATIVE_CUDA = True
    _SKIP_NATIVE_CUDA_REASON = str(_e)

@unittest.skipIf(_SKIP_NATIVE_CUDA, f"NativeVanillaAttentionCUDA not available: {_SKIP_NATIVE_CUDA_REASON}")
class TestNativeVanillaAttentionCUDA(unittest.TestCase):
    def setUp(self):
        self.attn = _native_cuda
        self.vanilla = VanillaAttention()

    def test_output_shape(self):
        for B, H, N, d in [(1, 1, 4, 4), (2, 2, 8, 8)]:
            with self.subTest(B=B, H=H, N=N, d=d):
                q, k, v = make_qkv(B, H, N, d, device="cuda")
                out = self.attn(q, k, v)
                self.assertEqual(out.shape, (B, H, N, d))

    def test_matches_vanilla(self):
        for B, H, N, d in [(1, 1, 4, 4), (2, 2, 32, 32), (1, 4, 128, 64)]:
            with self.subTest(B=B, H=H, N=N, d=d):
                q, k, v = make_qkv(B, H, N, d, device="cpu")
                out_native = self.attn(q, k, v).cpu()
                out_vanilla = self.vanilla(q, k, v)
                self.assertTrue(
                    torch.allclose(out_native, out_vanilla, atol=ATOL, rtol=RTOL),
                    msg=f"Max diff: {(out_native - out_vanilla).abs().max().item():.2e}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
