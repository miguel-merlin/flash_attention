"""
Bridge test: real GPT-2 Q/K/V -> SimplePagedKVCache -> paged_attention_v2_forward.

Goal
----
Validate that our ``paged_attention_v2_forward`` kernel produces the same
decode-step attention output as a plain PyTorch reference when fed real
activations pulled out of GPT-2 block 0.

This is specifically the "synthetic tests -> real model" bridge: everything
upstream of the kernel (embeddings, layer norm, c_attn, head splitting) is the
exact HF GPT-2 code path, so any numerical disagreement is attributable to the
kernel or the paged layout, not to a mismatched Q/K/V source.

Runs either as:
    pytest tests/test_gpt2_paged_attention_decode.py
or as a script:
    python3 tests/test_gpt2_paged_attention_decode.py

On a CPU-only box the CUDA-only checks are skipped but the paged cache
reconstruction check still runs, because ``SimplePagedKVCache`` is pure
PyTorch.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Lazy imports with graceful skip messages
# ---------------------------------------------------------------------------

def _import_simple_paged_kv_cache():
    try:
        from experiments.paged_kv_cache import SimplePagedKVCache  # type: ignore
        return SimplePagedKVCache
    except Exception as e:
        raise ImportError(f"Could not import SimplePagedKVCache: {e}") from e


def _import_paged_v2():
    """Returns (paged_attention_v2_forward, extension_loaded_bool)."""
    try:
        import flash_attn as fa  # type: ignore
        from flash_attn.ops import paged_attention_v2_forward  # type: ignore
        return paged_attention_v2_forward, bool(getattr(fa, "_EXTENSION_LOADED", False))
    except Exception as e:
        return None, False


def _import_gpt2():
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer  # type: ignore
        return GPT2LMHeadModel, GPT2Tokenizer
    except Exception as e:
        raise ImportError(f"transformers not available: {e}") from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """(B, N, H*d) -> (B, H, N, d)"""
    B, N, C = x.shape
    assert C == num_heads * head_dim, (
        f"split_heads got C={C} but expected H*d={num_heads * head_dim}"
    )
    return x.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def _torch_decode_attention(
    q_decode: torch.Tensor,     # (B, H, d)
    k_ref: torch.Tensor,        # (B, H, N, d)
    v_ref: torch.Tensor,        # (B, H, N, d)
) -> torch.Tensor:
    """
    Reference decode attention: q attends to all N cached positions.
    No causal mask needed because the cache only contains past tokens.
    Returns (B, H, d).
    """
    d = q_decode.shape[-1]
    scale = 1.0 / math.sqrt(d)
    # (B, H, 1, d) @ (B, H, d, N) -> (B, H, 1, N)
    scores = torch.matmul(q_decode.unsqueeze(-2), k_ref.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    # (B, H, 1, N) @ (B, H, N, d) -> (B, H, 1, d) -> (B, H, d)
    out = torch.matmul(probs, v_ref).squeeze(-2)
    return out


def _extract_gpt2_block0_qkv(model, input_ids: torch.Tensor):
    """
    Run embeddings -> block0.ln_1 -> block0.attn.c_attn by hand and return
    (q, k, v) in (B, H, N, d) shape plus dimensional metadata.
    """
    cfg = model.config
    H = cfg.n_head
    E = cfg.n_embd
    d = E // H

    transformer = model.transformer
    block0 = transformer.h[0]

    B, N = input_ids.shape
    device = input_ids.device

    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

    hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)
    x = block0.ln_1(hidden)
    qkv = block0.attn.c_attn(x)   # (B, N, 3E)

    q, k, v = qkv.split(E, dim=2)   # each (B, N, E)
    q = _split_heads(q, H, d)
    k = _split_heads(k, H, d)
    v = _split_heads(v, H, d)
    return q, k, v, H, d


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

class GPT2PagedAttentionDecodeTest(unittest.TestCase):
    """
    End-to-end bridge test using real GPT-2 activations.

    The test is split into two independent pieces so CPU-only environments
    can still exercise the paged cache layout:

      * ``test_cache_reconstruction`` runs everywhere — it only needs torch
        and transformers. It catches page-layout bugs before blaming the
        kernel.

      * ``test_paged_v2_matches_reference`` additionally requires CUDA and
        our compiled extension. On CPU it's skipped cleanly.
    """

    # Shared across both sub-tests
    PROMPT_IDS_LEN = 32

    @classmethod
    def _build_fixtures(cls, device: torch.device):
        GPT2LMHeadModel, _ = _import_gpt2()
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

        torch.manual_seed(0)
        vocab = model.config.vocab_size
        input_ids = torch.randint(
            low=0, high=vocab, size=(1, cls.PROMPT_IDS_LEN),
            dtype=torch.long, device=device,
        )

        with torch.no_grad():
            q, k, v, H, d = _extract_gpt2_block0_qkv(model, input_ids)

        # Cast to float32 for custom kernel compatibility.
        q = q.float()
        k = k.float()
        v = v.float()

        return model, input_ids, q, k, v, H, d

    # ------------------------------------------------------------------

    def test_cache_reconstruction(self):
        try:
            SimplePagedKVCache = _import_simple_paged_kv_cache()
        except ImportError as e:
            self.skipTest(str(e))

        try:
            _import_gpt2()
        except ImportError as e:
            self.skipTest(f"transformers/GPT-2 weights unavailable: {e}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[cache-reconstruction] device={device}")

        _, _, q, k, v, H, d = self._build_fixtures(device)
        B, _, N, _ = q.shape
        print(f"[cache-reconstruction] prompt_len={N}  q/k/v={tuple(q.shape)}")

        cache = SimplePagedKVCache(
            num_layers=1,
            num_heads=H,
            head_dim=d,
            max_seq_len=1024,
            page_size=16,
            batch_size=B,
            device=device,
            dtype=torch.float32,
        )
        cache.write_prefill(0, k, v)

        k_recon, v_recon = cache.reconstruct(0)
        self.assertEqual(tuple(k_recon.shape), (1, H, N, d))
        self.assertEqual(tuple(v_recon.shape), (1, H, N, d))

        k_err = (k_recon - k).abs().max().item()
        v_err = (v_recon - v).abs().max().item()
        print(f"[cache-reconstruction] max|k_recon - k| = {k_err:.3e}")
        print(f"[cache-reconstruction] max|v_recon - v| = {v_err:.3e}")

        # Round-tripping a float32 tensor through indexed writes must be exact.
        self.assertTrue(torch.allclose(k_recon, k, atol=0, rtol=0),
                        f"K reconstruction drift: {k_err}")
        self.assertTrue(torch.allclose(v_recon, v, atol=0, rtol=0),
                        f"V reconstruction drift: {v_err}")

    # ------------------------------------------------------------------

    def test_paged_v2_matches_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available; skipping custom paged_attention_v2_forward test.")

        paged_v2, ext_loaded = _import_paged_v2()
        if paged_v2 is None or not ext_loaded:
            self.skipTest(
                "flash_attn C++/CUDA extension not loaded; build it with "
                "'pip install -e . --no-build-isolation' and re-run on a CUDA machine."
            )

        try:
            SimplePagedKVCache = _import_simple_paged_kv_cache()
        except ImportError as e:
            self.skipTest(str(e))

        try:
            _import_gpt2()
        except ImportError as e:
            self.skipTest(f"transformers/GPT-2 weights unavailable: {e}")

        device = torch.device("cuda")
        print(f"[paged-v2] device={device}")

        _, _, q, k, v, H, d = self._build_fixtures(device)
        B, _, N, _ = q.shape
        print(f"[paged-v2] prompt_len={N}  q/k/v={tuple(q.shape)}")

        # Decode query = last token's Q, shape (B, H, d)
        q_decode = q[:, :, -1, :].contiguous()

        # Cache holds all N tokens (including the "last" position we treat as
        # already-decoded context). The reference attends over the same N keys.
        k_ref = k
        v_ref = v

        # Reference decode.
        out_ref = _torch_decode_attention(q_decode, k_ref, v_ref)
        self.assertEqual(tuple(out_ref.shape), (B, H, d))

        # Paged path.
        cache = SimplePagedKVCache(
            num_layers=1,
            num_heads=H,
            head_dim=d,
            max_seq_len=1024,
            page_size=16,
            batch_size=B,
            device=device,
            dtype=torch.float32,
        )
        cache.write_prefill(0, k, v)
        k_cache, v_cache, page_table, seq_lens = cache.get_layer_cache(0)
        print(f"[paged-v2] k_cache={tuple(k_cache.shape)}  "
              f"page_table={tuple(page_table.shape)}  seq_lens={seq_lens.tolist()}")

        out_paged = paged_v2(q_decode, k_cache, v_cache, page_table, seq_lens, False)
        self.assertEqual(tuple(out_paged.shape), (B, H, d))

        max_abs = (out_paged - out_ref).abs().max().item()
        mean_abs = (out_paged - out_ref).abs().mean().item()
        print(f"[paged-v2] max|paged - ref| = {max_abs:.3e}")
        print(f"[paged-v2] mean|paged - ref| = {mean_abs:.3e}")

        if not torch.allclose(out_paged, out_ref, atol=1e-3, rtol=1e-3):
            print(
                "[paged-v2] WARNING: outputs differ beyond atol/rtol=1e-3. "
                "Showing a few sample values for debugging:"
            )
            print("  out_ref  [0,0,:8] =", out_ref[0, 0, :8].tolist())
            print("  out_paged[0,0,:8] =", out_paged[0, 0, :8].tolist())
            self.fail(
                f"paged_attention_v2_forward output mismatches PyTorch reference: "
                f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}"
            )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """
    Script entry point. Accepts ``-h``/``--help`` and a ``-v``/``--verbose``
    toggle; otherwise runs the bridge test and prints a PASS/FAIL banner.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Bridge test: real GPT-2 Q/K/V -> SimplePagedKVCache -> "
            "paged_attention_v2_forward. Runs two sub-tests: cache "
            "reconstruction (CPU or CUDA) and paged v2 vs PyTorch reference "
            "(CUDA only; skipped cleanly on CPU)."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Verbose test runner output (default: on).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet test runner output (overrides --verbose).",
    )
    args = parser.parse_args(argv)

    verbosity = 0 if args.quiet else (2 if args.verbose else 1)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(GPT2PagedAttentionDecodeTest)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    print()
    if result.wasSuccessful():
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
