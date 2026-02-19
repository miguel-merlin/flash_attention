import torch
import flash_attn_cuda


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        O = flash_attn_cuda.flash_attention_forward(Q, K, V)
        ctx.save_for_backward(Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        # Implement the Flash Attention backward pass here
        # (recompute attention from saved Q, K, V — the key memory saving)
        raise NotImplementedError("Backward not yet implemented")


def flash_attention(Q, K, V):
    return FlashAttentionFunction.apply(Q, K, V)
