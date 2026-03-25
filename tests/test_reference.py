"""
Reference implementation parity tests.

Verifies that attention_reference_torch and attention_reference_manual agree
across a range of (B, H, N, d) shapes.

Usage:
    source .venv/bin/activate
    python3 tests/test_reference.py
  or via the full suite:
    python3 -m pytest tests/ -v
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.reference import attention_reference_torch, attention_reference_manual



class TestReferenceParity(unittest.TestCase):

    CASES = [
        (1, 1, 4, 3),
        (2, 2, 8, 4),
        (1, 4, 16, 8),
        (3, 1, 5, 7),
    ]

    def setUp(self):
        torch.manual_seed(0)

    def test_parity(self):
        for B, H, N, d in self.CASES:
            with self.subTest(B=B, H=H, N=N, d=d):
                q = torch.randn(B, H, N, d, dtype=torch.float32)
                k = torch.randn(B, H, N, d, dtype=torch.float32)
                v = torch.randn(B, H, N, d, dtype=torch.float32)

                out_torch  = attention_reference_torch(q, k, v)
                out_manual = attention_reference_manual(q, k, v)

                self.assertEqual(out_torch.shape, (B, H, N, d))
                self.assertEqual(out_manual.shape, (B, H, N, d))
                self.assertTrue(
                    torch.allclose(out_torch, out_manual, atol=1e-5, rtol=1e-5),
                    msg=f"Max diff: {(out_torch - out_manual).abs().max().item():.2e}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
