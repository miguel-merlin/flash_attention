"""
End-to-end GPT-2 generation TPS sanity benchmark.

This script benchmarks *full* Hugging Face ``model.generate(...)`` throughput
for GPT-2 with the standard attention kernel, and optionally again after
patching the attention module with our custom ``FlashAttentionCUDA`` kernel.

Why this benchmark exists:
    It is a coarse end-to-end sanity check — it measures the full generation
    loop (prefill + repeated decode + sampling bookkeeping), not isolated
    attention cost. FlashAttention targets prefill/full-attention, so
    end-to-end numbers will typically show only a small delta.

For isolated attention-kernel measurements see:
    * benchmarks/bench_gpt2_prefill.py — prompt/prefill phase only
    * benchmarks/bench_gpt2_decode.py  — single-token decode with KV cache

Usage:
    source .venv/bin/activate
    python3 benchmarks/bench_tps.py --tokens 50 --iters 5 --print-output
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import types

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def patch_gpt2_attention(model, custom_attn_module) -> None:
    """
    Patch every GPT-2 block's ``_attn`` method to use ``custom_attn_module``.

    The custom module is expected to accept ``(q, k, v, is_causal=True)`` and
    return the attention output with the same (B, H, N, d) layout GPT-2 uses
    internally.
    """
    for block in model.transformer.h:
        def custom_attn_forward(self, query, key, value, attention_mask=None, head_mask=None):
            out = custom_attn_module(query, key, value, is_causal=True)
            return out, None
        block.attn._attn = types.MethodType(custom_attn_forward, block.attn)


def sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_tps(
    model,
    inputs,
    tokenizer,
    max_new_tokens: int,
    iters: int,
) -> tuple[float, str]:
    """
    Run ``model.generate(...)`` ``iters`` times and return:
        (tokens_per_second, decoded_text_from_last_iter)

    Tokens per second is computed against the *median* wall-clock time across
    iterations to reduce noise. ``do_sample=False`` keeps results deterministic.
    """
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        _ = model.generate(**inputs, **gen_kwargs)

    times: list[float] = []
    last_output_ids = None
    for _ in range(iters):
        sync_if_cuda()
        start = time.perf_counter()
        with torch.no_grad():
            last_output_ids = model.generate(**inputs, **gen_kwargs)
        sync_if_cuda()
        end = time.perf_counter()
        times.append(end - start)

    times.sort()
    median_time = times[len(times) // 2]
    tps = max_new_tokens / median_time

    decoded = tokenizer.decode(last_output_ids[0], skip_special_tokens=True) if last_output_ids is not None else ""
    return tps, decoded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end GPT-2 generation TPS sanity benchmark"
    )
    parser.add_argument("--tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--iters", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The Hugging Face open source models are",
        help="Input prompt",
    )
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="Print generated text for standard and patched GPT-2",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 64)
    print("End-to-end GPT-2 generation TPS")
    print("=" * 64)
    print(f"Device: {device}")
    if device.type == "cpu":
        print(
            "WARNING: running on CPU. End-to-end GPT-2 generation will be slow "
            "and the custom FlashAttention CUDA benchmark will be skipped."
        )

    print("Loading GPT-2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    print(f"\nBenchmarking generation of {args.tokens} new tokens ({args.iters} iters)...")

    # Standard GPT-2
    print("\n[ Standard GPT-2 Attention ]")
    std_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    std_model.eval()
    standard_tps, standard_text = measure_tps(
        std_model, inputs, tokenizer, max_new_tokens=args.tokens, iters=args.iters
    )
    print(f"TPS: {standard_tps:.2f} tokens/sec")
    if args.print_output:
        print("--- standard output ---")
        print(standard_text)
        print("-----------------------")

    # Release standard model before loading/patching the next one (helps on small GPUs)
    del std_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Patched GPT-2 (CUDA only)
    if device.type == "cuda":
        try:
            from flash_attn.attention import FlashAttentionCUDA

            print("\n[ Patched with Custom FlashAttentionCUDA ]")
            patched_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
            patched_model.eval()
            custom_attn = FlashAttentionCUDA(device=device)
            patch_gpt2_attention(patched_model, custom_attn)

            patched_tps, patched_text = measure_tps(
                patched_model, inputs, tokenizer, max_new_tokens=args.tokens, iters=args.iters
            )
            print(f"TPS: {patched_tps:.2f} tokens/sec")
            if args.print_output:
                print("--- patched output ---")
                print(patched_text)
                print("----------------------")

            speedup = patched_tps / standard_tps
            print(f"\nSpeedup (patched / standard): {speedup:.2f}x")
        except Exception as e:
            print(f"Could not benchmark custom FlashAttention CUDA: {e}")
    else:
        print("\n[ Skipping FlashAttentionCUDA benchmark — CUDA not available ]")

    print(
        "\nNote: this benchmark measures full Hugging Face generation, "
        "not isolated prefill or decode attention."
    )


if __name__ == "__main__":
    main()
