"""
flash_attn package

Exports:
    VanillaAttention      - pure-PyTorch attention (CPU & CUDA, fully differentiable)
    FlashAttentionCPP     - C++ CPU kernel via our custom extension
    FlashAttentionCUDA    - CUDA kernel via our custom extension (placeholder)
    flash_attention_forward - raw op dispatch helper
"""

# Try to load the compiled C extension; skip gracefully if not yet built.
# Importing _C triggers TORCH_LIBRARY registration in the .so, which must
# happen BEFORE we call register_ops() (which registers the fake impl and
# autograd hooks for the now-existing op).
try:
    from . import _C  # noqa: F401
    _EXTENSION_LOADED = True
except ImportError:
    _EXTENSION_LOADED = False

from .ops import (
    flash_attention_forward,
    paged_attention_forward,
    register_ops,
    register_paged_ops,
)
from .attention import VanillaAttention, FlashAttentionCPP, FlashAttentionCUDA
from .reference import attention_reference_torch, attention_reference_manual

# Register fake impl + autograd only when the extension is present
if _EXTENSION_LOADED:
    register_ops()
    register_paged_ops()

__all__ = [
    "VanillaAttention",
    "FlashAttentionCPP",
    "FlashAttentionCUDA",
    "flash_attention_forward",
    "paged_attention_forward",
    "attention_reference_torch",
    "attention_reference_manual",
]

__version__ = "0.1.0"