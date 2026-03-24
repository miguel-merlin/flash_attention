import torch
from flash_attn.reference import attention_reference_torch, attention_reference_manual


def run_test(B, H, N, d):
    print(f"\nRunning test with B={B}, H={H}, N={N}, d={d}")

    q = torch.randn(B, H, N, d, dtype=torch.float32)
    k = torch.randn(B, H, N, d, dtype=torch.float32)
    v = torch.randn(B, H, N, d, dtype=torch.float32)

    out_torch = attention_reference_torch(q, k, v)
    out_manual = attention_reference_manual(q, k, v)

    # Check shape correctness
    assert out_torch.shape == (B, H, N, d)
    assert out_manual.shape == (B, H, N, d)

    diff = torch.max(torch.abs(out_torch - out_manual)).item()

    #print("output shape:", out_torch.shape)
    print("max abs diff:", diff)
    print("allclose:", torch.allclose(out_torch, out_manual, atol=1e-5, rtol=1e-5))

    assert torch.allclose(out_torch, out_manual, atol=1e-5, rtol=1e-5)


def main():
    torch.manual_seed(0)

    test_cases = [
        (1, 1, 4, 3),
        (2, 2, 8, 4),
        (1, 4, 16, 8),
        (3, 1, 5, 7),
    ]

    for B, H, N, d in test_cases:
        run_test(B, H, N, d)

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
