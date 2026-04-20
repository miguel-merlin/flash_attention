"""
Minimal GPT-2 generation prototype with Flash-prefill + Paged-decode.

This is an *educational* runner. It deliberately re-implements GPT-2's forward
pass in Python so that:

  * the prefill step goes through our ``FlashAttentionCUDA`` kernel, and
  * each decode step goes through ``paged_attention_v2_forward`` against a
    ``SimplePagedKVCache``.

The goal is correctness-on-real-GPT-2 and a readable reference, not speed.
Batch size is hard-coded to 1, decoding is greedy, and all attention math is
done one layer at a time.

Modes
-----
  torch       : manual GPT-2 forward using pure PyTorch attention (reference).
  flash-paged : FlashAttention prefill + PagedAttention v2 decode (custom kernels).
  compare     : run both and print both outputs.

Examples
--------
  python3 experiments/gpt2_flash_paged_generate.py --mode torch --tokens 20
  python3 experiments/gpt2_flash_paged_generate.py --mode flash-paged --tokens 20
  python3 experiments/gpt2_flash_paged_generate.py --mode compare --tokens 20
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.paged_kv_cache import SimplePagedKVCache  # noqa: E402


# ---------------------------------------------------------------------------
# Head reshape helpers
# ---------------------------------------------------------------------------

def split_heads(x: torch.Tensor, B: int, N: int, H: int, d: int) -> torch.Tensor:
    """(B, N, H*d) -> (B, H, N, d)"""
    return x.view(B, N, H, d).permute(0, 2, 1, 3).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """
    (B, H, N, d) -> (B, N, H*d)
    Works for N==1 (decode) as well as long prefill.
    """
    B, H, N, d = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, N, H * d)


# ---------------------------------------------------------------------------
# Attention references
# ---------------------------------------------------------------------------

def torch_full_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """
    Pure-PyTorch scaled dot-product attention over a full prompt.

    Shapes:
        q, k, v: (B, H, N, d)
        returns: (B, H, N, d)
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, H, N, N)
    if causal:
        N = scores.shape[-1]
        mask = torch.triu(
            torch.ones((N, N), dtype=torch.bool, device=scores.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def torch_decode_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    Reference one-step decode attention.

    Shapes:
        q: (B, H, d)     — query for the new token only
        k: (B, H, N, d)  — cached keys for all previous tokens incl. current
        v: (B, H, N, d)
        returns: (B, H, d)
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    # (B, H, 1, d) @ (B, H, d, N) -> (B, H, 1, N)
    scores = torch.matmul(q.unsqueeze(-2), k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v).squeeze(-2)
    return out


# ---------------------------------------------------------------------------
# Lazy custom-op hooks (only needed for flash-paged)
# ---------------------------------------------------------------------------

def _load_flash_paged_ops():
    """
    Import FlashAttentionCUDA + paged_attention_v2_forward.

    Returns ``(FlashAttentionCUDA_instance, paged_attention_v2_forward)`` or
    raises RuntimeError with a clear message.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("flash-paged mode requires CUDA and built custom extension.")
    try:
        import flash_attn as fa
        if not getattr(fa, "_EXTENSION_LOADED", False):
            raise RuntimeError(
                "flash_attn C++/CUDA extension not loaded. "
                "Build it with: pip install -e . --no-build-isolation"
            )
        from flash_attn import FlashAttentionCUDA
        from flash_attn.ops import paged_attention_v2_forward
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Could not load custom flash/paged ops: {e}") from e

    return FlashAttentionCUDA(), paged_attention_v2_forward


# ---------------------------------------------------------------------------
# Per-block forward helpers
# ---------------------------------------------------------------------------

def block_prefill(
    block,
    hidden: torch.Tensor,
    *,
    H: int,
    d: int,
    mode: str,
    layer_idx: int,
    paged_cache: Optional[SimplePagedKVCache],
    contiguous_cache: Optional[Dict[int, Dict[str, torch.Tensor]]],
    flash_module,
) -> torch.Tensor:
    """
    Run one GPT-2 block over the full prompt and, as a side effect, populate
    either the paged cache (flash-paged mode) or the contiguous cache (torch
    mode) for ``layer_idx``.

    ``hidden`` is (B, N, E); the returned hidden has the same shape.
    """
    B, N, E = hidden.shape

    # Attention sub-block.
    residual = hidden
    x = block.ln_1(hidden)
    qkv = block.attn.c_attn(x)  # (B, N, 3E)
    q, k, v = qkv.split(E, dim=-1)
    q = split_heads(q, B, N, H, d)
    k = split_heads(k, B, N, H, d)
    v = split_heads(v, B, N, H, d)

    if mode == "flash-paged":
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        attn_out = flash_module(q_f, k_f, v_f, is_causal=True)
        attn_out = attn_out.to(hidden.dtype)
        assert paged_cache is not None
        # Cache is stored in float32 to match the kernel's expectation.
        paged_cache.write_prefill(layer_idx, k_f, v_f)
    else:
        attn_out = torch_full_attention(q, k, v, causal=True)
        assert contiguous_cache is not None
        contiguous_cache[layer_idx] = {
            "k": k.detach(),
            "v": v.detach(),
        }

    attn_out = merge_heads(attn_out)                     # (B, N, E)
    attn_out = block.attn.c_proj(attn_out)
    hidden = residual + attn_out

    # MLP sub-block.
    residual = hidden
    x = block.ln_2(hidden)
    mlp_out = block.mlp(x)
    hidden = residual + mlp_out
    return hidden


def block_decode(
    block,
    hidden: torch.Tensor,
    *,
    H: int,
    d: int,
    mode: str,
    layer_idx: int,
    paged_cache: Optional[SimplePagedKVCache],
    contiguous_cache: Optional[Dict[int, Dict[str, torch.Tensor]]],
    paged_v2_fn,
    position: Optional[int] = None,
) -> torch.Tensor:
    """
    Run one GPT-2 block for a single decode token.

    ``hidden`` is (B, 1, E); returned hidden has the same shape.
    The caches are updated in place (appended) so subsequent tokens see the
    new K/V.
    """
    B, one, E = hidden.shape
    assert one == 1, f"block_decode expects N=1, got {one}"

    residual = hidden
    x = block.ln_1(hidden)
    qkv = block.attn.c_attn(x)  # (B, 1, 3E)
    q, k, v = qkv.split(E, dim=-1)   # each (B, 1, E)

    # (B, 1, E) -> (B, H, 1, d) -> (B, H, d)
    q_new = split_heads(q, B, 1, H, d).squeeze(2)
    k_new = split_heads(k, B, 1, H, d).squeeze(2)
    v_new = split_heads(v, B, 1, H, d).squeeze(2)

    if mode == "flash-paged":
        assert paged_cache is not None
        # Pass an explicit ``position`` so all GPT-2 layers in this decode
        # step write their K/V at the same token position. seq_lens is a
        # single tensor shared across layers; without this, each layer
        # would advance it and later layers would overwrite the wrong slot.
        paged_cache.append_decode(
            layer_idx, k_new.float(), v_new.float(), position=position,
        )
        k_cache, v_cache, page_table, seq_lens = paged_cache.get_layer_cache(layer_idx)
        out_bhd = paged_v2_fn(
            q_new.float(), k_cache, v_cache, page_table, seq_lens, False
        )  # (B, H, d)
        out_bhd = out_bhd.to(hidden.dtype)
    else:
        assert contiguous_cache is not None
        k_prev = contiguous_cache[layer_idx]["k"]           # (B, H, N, d)
        v_prev = contiguous_cache[layer_idx]["v"]
        k_full = torch.cat([k_prev, k_new.unsqueeze(2)], dim=2)
        v_full = torch.cat([v_prev, v_new.unsqueeze(2)], dim=2)
        contiguous_cache[layer_idx] = {"k": k_full, "v": v_full}
        out_bhd = torch_decode_attention(q_new, k_full, v_full)   # (B, H, d)

    # (B, H, d) -> (B, H, 1, d) -> (B, 1, E)
    out_bhnd = out_bhd.unsqueeze(2)
    out = merge_heads(out_bhnd)      # (B, 1, E)
    out = block.attn.c_proj(out)
    hidden = residual + out

    residual = hidden
    x = block.ln_2(hidden)
    mlp_out = block.mlp(x)
    hidden = residual + mlp_out
    return hidden


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------

def _embed(model, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
    """
    Return wte(input_ids) + wpe(positions). ``start_pos`` is the absolute
    sequence position of ``input_ids[:, 0]``.
    """
    B, N = input_ids.shape
    device = input_ids.device
    positions = torch.arange(start_pos, start_pos + N, dtype=torch.long, device=device)
    positions = positions.unsqueeze(0).expand(B, -1)
    return model.transformer.wte(input_ids) + model.transformer.wpe(positions)


@torch.no_grad()
def generate(
    *,
    model,
    tokenizer,
    prompt: str,
    tokens: int,
    mode: str,
    max_seq_len: int,
    page_size: int,
    device: torch.device,
) -> Tuple[List[int], str]:
    """
    Run manual GPT-2 generation under the requested mode. Returns
    ``(token_ids, decoded_text)`` for the newly generated tokens.

    Prefill and decode use the same block helpers; the only difference is
    which cache and which attention function they drive.
    """
    cfg = model.config
    H = cfg.n_head
    E = cfg.n_embd
    d = E // H
    L = cfg.n_layer

    # Prepare caches.
    paged_cache: Optional[SimplePagedKVCache] = None
    contiguous_cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    flash_module = None
    paged_v2_fn = None

    if mode == "flash-paged":
        flash_module, paged_v2_fn = _load_flash_paged_ops()
        paged_cache = SimplePagedKVCache(
            num_layers=L,
            num_heads=H,
            head_dim=d,
            max_seq_len=max_seq_len,
            page_size=page_size,
            batch_size=1,
            device=device,
            dtype=torch.float32,
        )
    else:
        contiguous_cache = {}

    # Tokenize and run prefill.
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    B, N_prompt = input_ids.shape
    assert B == 1, "This prototype only supports batch size 1."
    if N_prompt > max_seq_len:
        raise ValueError(
            f"Prompt length {N_prompt} exceeds --max-seq-len {max_seq_len}."
        )
    if N_prompt + tokens > max_seq_len:
        raise ValueError(
            f"Prompt ({N_prompt}) + generated ({tokens}) > --max-seq-len "
            f"({max_seq_len}); increase --max-seq-len."
        )

    hidden = _embed(model, input_ids, start_pos=0)
    for layer_idx, block in enumerate(model.transformer.h):
        hidden = block_prefill(
            block,
            hidden,
            H=H,
            d=d,
            mode=mode,
            layer_idx=layer_idx,
            paged_cache=paged_cache,
            contiguous_cache=contiguous_cache,
            flash_module=flash_module,
        )

    hidden = model.transformer.ln_f(hidden)
    logits = model.lm_head(hidden)  # (1, N_prompt, V)
    next_token = torch.argmax(logits[:, -1, :], dim=-1)   # (1,)

    generated: List[int] = [int(next_token.item())]
    current_len = N_prompt  # not yet including ``next_token`` as an input

    for step in range(tokens - 1):
        # Feed the just-generated token through the network.
        token_in = next_token.view(1, 1)  # (1, 1)
        hidden = _embed(model, token_in, start_pos=current_len)

        # All layers in this decode step write their K/V at the same
        # token position. The first layer is what actually advances
        # seq_lens in the paged cache; subsequent layers only fill their
        # own slot at this position.
        step_position = current_len

        for layer_idx, block in enumerate(model.transformer.h):
            hidden = block_decode(
                block,
                hidden,
                H=H,
                d=d,
                mode=mode,
                layer_idx=layer_idx,
                paged_cache=paged_cache,
                contiguous_cache=contiguous_cache,
                paged_v2_fn=paged_v2_fn,
                position=step_position,
            )

        hidden = model.transformer.ln_f(hidden)
        logits = model.lm_head(hidden)  # (1, 1, V)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        generated.append(int(next_token.item()))
        current_len += 1

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(
        description="Manual GPT-2 generation with optional flash-prefill + paged-decode."
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="The Hugging Face open source models are",
        help="Input prompt text.",
    )
    p.add_argument("--tokens", type=int, default=20, help="Number of tokens to generate.")
    p.add_argument("--max-seq-len", type=int, default=1024, help="Paged cache capacity per batch row.")
    p.add_argument("--page-size", type=int, default=16, help="Tokens per KV page.")
    p.add_argument("--device", type=str, default=default_device, help="torch device string.")
    p.add_argument(
        "--mode",
        type=str,
        choices=["torch", "flash-paged", "compare"],
        default="torch",
        help="Which attention path to drive.",
    )
    return p


def _load_model(device: torch.device):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    return model, tokenizer


def _run_single_mode(
    args,
    mode: str,
    device: torch.device,
    *,
    model=None,
    tokenizer=None,
    allow_exit: bool = True,
) -> bool:
    """
    Run one generation pass. Returns True on success, False on known failure
    (e.g. flash-paged requested but CUDA/extension missing). When
    ``allow_exit`` is True, known failures call ``sys.exit(1)`` instead of
    returning — this preserves the old single-mode CLI behavior while letting
    compare mode keep going.
    """
    if model is None or tokenizer is None:
        try:
            model, tokenizer = _load_model(device)
        except Exception as e:
            print(f"[ERROR] Could not load GPT-2: {e}")
            if allow_exit:
                sys.exit(1)
            return False

    try:
        ids, text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            tokens=args.tokens,
            mode=mode,
            max_seq_len=args.max_seq_len,
            page_size=args.page_size,
            device=device,
        )
    except RuntimeError as e:
        msg = str(e)
        if "flash-paged mode requires CUDA" in msg or "extension not loaded" in msg:
            print(f"[{mode}] {msg}")
            if allow_exit:
                sys.exit(1)
            return False
        raise

    print(f"[{mode}] prompt  : {args.prompt!r}")
    print(f"[{mode}] tokens  : {args.tokens}")
    print(f"[{mode}] new ids : {ids}")
    print(f"[{mode}] text    : {text!r}")
    print(f"[{mode}] full    : {args.prompt + text!r}")
    return True


def main() -> None:
    args = _build_arg_parser().parse_args()
    device = torch.device(args.device)

    print(f"device       : {device}")
    print(f"mode         : {args.mode}")
    print(f"max_seq_len  : {args.max_seq_len}")
    print(f"page_size    : {args.page_size}")
    print()

    if args.mode == "compare":
        try:
            model, tokenizer = _load_model(device)
        except Exception as e:
            print(f"[ERROR] Could not load GPT-2: {e}")
            sys.exit(1)
        _run_single_mode(args, "torch", device,
                         model=model, tokenizer=tokenizer, allow_exit=False)
        print()
        _run_single_mode(args, "flash-paged", device,
                         model=model, tokenizer=tokenizer, allow_exit=False)
    else:
        _run_single_mode(args, args.mode, device)


if __name__ == "__main__":
    main()
