"""
GPT-2 Attention Kernel Comparison UI
=====================================

A Gradio interface that runs GPT-2 text generation using different attention
kernel backends and displays tokens-per-second comparisons.

Backends:
  - PyTorch reference  (manual forward, pure-torch attention)
  - Flash + Paged      (FlashAttention prefill + PagedAttention v2 decode)

Usage:
    pip install gradio
    python app.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import gradio as gr

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.gpt2_flash_paged_generate import generate  # noqa: E402

# ---------------------------------------------------------------------------
# Global model singleton (loaded once)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_device = None


def _get_model():
    global _model, _tokenizer, _device
    if _model is None:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _tokenizer.pad_token = _tokenizer.eos_token
        _model = GPT2LMHeadModel.from_pretrained("gpt2").to(_device).eval()
    return _model, _tokenizer, _device


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    mode: str
    label: str
    text: str
    tokens: int
    elapsed: float
    tps: float
    error: Optional[str] = None


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _run_mode(
    model,
    tokenizer,
    prompt: str,
    tokens: int,
    mode: str,
    label: str,
    device: torch.device,
    max_seq_len: int,
    page_size: int,
) -> RunResult:
    try:
        # Warmup
        _sync(device)
        generate(
            model=model, tokenizer=tokenizer, prompt=prompt, tokens=min(tokens, 5),
            mode=mode, max_seq_len=max_seq_len, page_size=page_size, device=device,
        )
        _sync(device)

        # Timed run
        _sync(device)
        t0 = time.perf_counter()
        _ids, text = generate(
            model=model, tokenizer=tokenizer, prompt=prompt, tokens=tokens,
            mode=mode, max_seq_len=max_seq_len, page_size=page_size, device=device,
        )
        _sync(device)
        elapsed = time.perf_counter() - t0

        tps = tokens / elapsed if elapsed > 0 else 0.0
        return RunResult(
            mode=mode, label=label, text=text,
            tokens=tokens, elapsed=elapsed, tps=tps,
        )
    except Exception as e:
        return RunResult(
            mode=mode, label=label, text="",
            tokens=tokens, elapsed=0.0, tps=0.0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Main generation + comparison
# ---------------------------------------------------------------------------

MODES = [
    ("torch", "PyTorch Reference"),
    ("flash-paged", "Flash + Paged (Custom CUDA)"),
]


def run_comparison(
    prompt: str,
    max_new_tokens: int,
    max_seq_len: int,
    page_size: int,
):
    if not prompt or not prompt.strip():
        prompt = "The future of artificial intelligence is"

    max_new_tokens = int(max_new_tokens)
    max_seq_len = int(max_seq_len)
    page_size = int(page_size)

    model, tokenizer, device = _get_model()

    prompt_len = len(tokenizer.encode(prompt))
    if prompt_len + max_new_tokens > max_seq_len:
        max_seq_len = prompt_len + max_new_tokens + 32

    results: List[RunResult] = []
    for mode, label in MODES:
        r = _run_mode(
            model, tokenizer, prompt, max_new_tokens,
            mode, label, device, max_seq_len, page_size,
        )
        results.append(r)

    # --- Build outputs ---

    # Generated text panels
    text_outputs = []
    for r in results:
        if r.error:
            text_outputs.append(f"Error: {r.error}")
        else:
            text_outputs.append(prompt + r.text)

    # Metrics table
    table_rows = []
    for r in results:
        if r.error:
            table_rows.append([r.label, "Error", "---", "---"])
        else:
            table_rows.append([
                r.label,
                f"{r.elapsed * 1000:.1f} ms",
                f"{r.tps:.1f}",
                f"{r.elapsed / max_new_tokens * 1000:.1f} ms",
            ])

    # Bar chart data for TPS
    tps_chart = {r.label: r.tps for r in results if not r.error}

    # Speedup summary
    ok = [r for r in results if not r.error]
    summary_parts = []
    summary_parts.append(f"Device: {device}  |  Prompt tokens: {prompt_len}  |  Generated: {max_new_tokens}")
    if len(ok) == 2:
        r0, r1 = ok
        if r1.tps > 0 and r0.tps > 0:
            ratio = r0.tps / r1.tps
            if ratio > 1:
                summary_parts.append(f"{r0.label} is {ratio:.2f}x faster")
            else:
                summary_parts.append(f"{r1.label} is {1/ratio:.2f}x faster")
    summary = "\n".join(summary_parts)

    return (
        text_outputs[0] if len(text_outputs) > 0 else "",
        text_outputs[1] if len(text_outputs) > 1 else "",
        table_rows,
        tps_chart,
        summary,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="GPT-2 Attention Kernel Comparison",
    ) as app:
        gr.Markdown(
            "# GPT-2 Attention Kernel Comparison\n"
            "Compare **PyTorch reference** attention vs **Flash + Paged** custom CUDA kernels "
            "running inside a real GPT-2 model. See generated text and tokens/sec side by side."
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt_box = gr.Textbox(
                    label="Prompt",
                    value="The future of artificial intelligence is",
                    lines=2,
                    placeholder="Enter your prompt here...",
                )
            with gr.Column(scale=1):
                tokens_slider = gr.Slider(
                    minimum=5, maximum=100, value=20, step=5,
                    label="Tokens to generate",
                )
                max_seq_slider = gr.Slider(
                    minimum=128, maximum=2048, value=1024, step=64,
                    label="Max sequence length",
                )
                page_size_slider = gr.Slider(
                    minimum=4, maximum=64, value=16, step=4,
                    label="Page size",
                )

        run_btn = gr.Button("Generate & Compare", variant="primary", size="lg")

        summary_box = gr.Textbox(label="Summary", interactive=False)

        gr.Markdown("### Generated Text")
        with gr.Row():
            text_torch = gr.Textbox(
                label="PyTorch Reference",
                lines=5,
                interactive=False,
            )
            text_flash = gr.Textbox(
                label="Flash + Paged (Custom CUDA)",
                lines=5,
                interactive=False,
            )

        gr.Markdown("### Performance")
        with gr.Row():
            with gr.Column():
                metrics_table = gr.Dataframe(
                    headers=["Kernel", "Total Time", "Tokens/sec", "Time/token"],
                    label="Metrics",
                    interactive=False,
                )
            with gr.Column():
                tps_chart = gr.BarPlot(
                    x="Kernel",
                    y="Tokens/sec",
                    title="Tokens per Second",
                    height=300,
                )

        def _on_click(prompt, tokens, max_seq, page_size):
            t0, t1, table, tps_data, summary = run_comparison(
                prompt, tokens, max_seq, page_size,
            )
            # Convert tps dict to a dataframe for the bar plot
            import pandas as pd
            chart_df = pd.DataFrame([
                {"Kernel": k, "Tokens/sec": v} for k, v in tps_data.items()
            ])
            return t0, t1, table, chart_df, summary

        run_btn.click(
            fn=_on_click,
            inputs=[prompt_box, tokens_slider, max_seq_slider, page_size_slider],
            outputs=[text_torch, text_flash, metrics_table, tps_chart, summary_box],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, theme=gr.themes.Soft())
