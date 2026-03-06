import torch
from torch import Tensor

__all__ = ["flash_attention", "flash_attention_out"]


@torch.library.register_fake("extension_cpp::flash_attention")
def _(q, k, v) -> Tensor:
    torch._check(q.shape == k.shape == v.shape, "Input shapes must be the same")
    torch._check(q.dtype == torch.float16, "Input dtypes must be float16")
    torch._check(q.dtype == k.dtype == v.dtype, "Input dtypes must be the same")
    torch._check(q.device == k.device == v.device, "Input devices must be the same" )
    return torch.empty_like(q)

def _backward(ctx, grad):
    q, k, v = ctx.saved_tensors
    torch._check(q.shape == k.shape == v.shape, "Input shapes must be the same")
    torch._check(q.dtype == torch.float16, "Input dtypes must be float16")
    torch._check(q.dtype == k.dtype == v.dtype, "Input dtypes must be the same")
    torch._check(q.device == k.device == v.device, "Input devices must be the same" )
    return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))

def _setup_context(ctx, inputs, outputs):
    q, k, v = inputs
    saved_q, saved_k, saved_v = None, None, None
    if ctx.needs_input_grad[0]:
        saved_q = q
    if ctx.needs_input_grad[1]:
        saved_k = k
    if ctx.needs_input_grad[2]:
        saved_v = v
    ctx.save_for_backward(saved_q, saved_k, saved_v)

torch.library.register_autograd(
    "extension_cpp::flash_attention_forward", _backward, setup_context=_setup_context
)

@torch.library.register_fake("extension_cpp::flash_attention")
def _(q, k, v) -> Tensor:
    torch._check(q.shape == k.shape == v.shape, "Input shapes must be the same")
    torch._check(q.dtype == torch.float16, "Input dtypes must be float16")
    torch._check(q.dtype == k.dtype == v.dtype, "Input dtypes must be the same")
    torch._check(q.device == k.device == v.device, "Input devices must be the same" )
    return torch.empty_like(q)

def flash_attention_out(q: Tensor, k: Tensor, v: Tensor, out: Tensor) -> None:
    "Writes attention output to the out tensor."
    torch.ops.extension_cpp.flash_attention.default(q, k, v, out)