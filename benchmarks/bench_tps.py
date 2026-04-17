"""
Tokens per second benchmark for attention variants.

Usage:
    source .venv/bin/activate
    python3 benchmarks/bench_tps.py

Measures generation speed in Tokens Per Second (TPS) using a Hugging Face LLM (GPT-2).
"""

from __future__ import annotations

import argparse
import sys
import os
import time

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import types

from flash_attn.attention import FlashAttentionCUDA

def patch_gpt2_attention(model, custom_attn_module):
    """
    Patches the given GPT2 model to use the custom_attn_module.
    """
    for block in model.transformer.h:
        def custom_attn_forward(self, query, key, value, attention_mask=None, head_mask=None):
            out = custom_attn_module(query, key, value, is_causal=True)
            return out, None
        block.attn._attn = types.MethodType(custom_attn_forward, block.attn)

def measure_tps(model, inputs, max_new_tokens: int, iters: int = 3) -> float:
    """
    Generate tokens and calculate the median tokens per second.
    """
    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
    times = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        
    # calculate total generated tokens
    # Note: batch size is 1 here
    total_tokens = max_new_tokens 
    
    # Calculate median time
    times.sort()
    median_time = times[len(times) // 2]
    
    return total_tokens / median_time

def main():
    parser = argparse.ArgumentParser(description="Flash Attention TPS benchmark")
    parser.add_argument("--tokens", type=int, default=1_000, help="Number of tokens to generate")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--prompt", type=str, default="The Hugging Face open source models are", help="Input prompt")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running TPS benchmark on device: {device}")
    
    print("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    # We use a clean copy of the model for standard and later patch it
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    inputs = tokenizer(args.prompt, return_tensors='pt').to(device)
    
    print(f"\nBenchmarking generation of {args.tokens} new tokens...")

    # 1. Standard Attention
    print("\n[ Standard GPT-2 Attention ]")
    standard_tps = measure_tps(model, inputs, max_new_tokens=args.tokens, iters=args.iters)
    print(f"TPS: {standard_tps:.2f} tokens/sec")

    # 2. Patch with Custom Attention
    if device.type == 'cuda':
        try:
            print("\n[ Patching with Custom Flash Attention CUDA ]")
            custom_attn = FlashAttentionCUDA(device=device)
            patch_gpt2_attention(model, custom_attn)
            
            patched_tps = measure_tps(model, inputs, max_new_tokens=args.tokens, iters=args.iters)
            print(f"TPS: {patched_tps:.2f} tokens/sec")
            
            speedup = patched_tps / standard_tps
            print(f"\nSpeedup: {speedup:.2f}x")
        except Exception as e:
            print(f"Could not benchmark custom Flash Attention CUDA: {e}")
    else:
        print("\n[ Skipping Flash Attention CUDA test (requires GPU) ]")

if __name__ == "__main__":
    main()
