"""
Reference attention implementations for correctness testing.

Both implementations expect CPU float32 tensors with shape (B, H, N, d):
  B - batch size
  H - number of heads
  N - sequence length
  d - head dimension
"""

import math
import torch


def attention_reference_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Naive reference attention implementation using PyTorch ops.

    Input shape:
        q, k, v: (B, H, N, d)

    Output shape:
        (B, H, N, d)
    """
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError("q, k, v must all have the same shape")

    if q.dim() != 4:
        raise ValueError("q, k, v must have shape (B, H, N, d)")

    if q.device.type != "cpu" or k.device.type != "cpu" or v.device.type != "cpu":
        raise ValueError("reference implementation expects CPU tensors")

    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("reference implementation expects float32 tensors")

    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    # Softmax across key positions
    probs = torch.softmax(scores, dim=-1)

    # (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
    out = torch.matmul(probs, v)

    return out


def attention_reference_manual(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Fully manual CPU attention implementation using explicit Python loops.

    Input shape:
        q, k, v: (B, H, N, d)

    Output shape:
        (B, H, N, d)
    """
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError("q, k, v must all have the same shape")

    if q.dim() != 4:
        raise ValueError("q, k, v must have shape (B, H, N, d)")

    if q.device.type != "cpu" or k.device.type != "cpu" or v.device.type != "cpu":
        raise ValueError("manual reference expects CPU tensors")

    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("manual reference expects float32 tensors")

    B, H, N, d = q.shape
    scale = 1.0 / math.sqrt(d)

    out = torch.zeros_like(q)

    for b in range(B):
        for h in range(H):
            scores = torch.empty((N, N), dtype=torch.float32)

            for i in range(N):
                for j in range(N):
                    dot = 0.0
                    for x in range(d):
                        dot += q[b, h, i, x].item() * k[b, h, j, x].item()
                    scores[i, j] = dot * scale

            probs = torch.empty((N, N), dtype=torch.float32)

            for i in range(N):
                row = scores[i]
                row_max = torch.max(row).item()

                exp_sum = 0.0
                for j in range(N):
                    exp_sum += math.exp(row[j].item() - row_max)

                for j in range(N):
                    probs[i, j] = math.exp(row[j].item() - row_max) / exp_sum

            for i in range(N):
                for x in range(d):
                    acc = 0.0
                    for j in range(N):
                        acc += probs[i, j].item() * v[b, h, j, x].item()
                    out[b, h, i, x] = acc

    return out
