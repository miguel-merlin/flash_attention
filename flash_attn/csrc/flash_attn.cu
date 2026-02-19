#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

torch::Tensor flash_attention_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    CHECK_CUDA(Q);
    CHECK_CONTIGUOUS(Q);
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), d = Q.size(3);
    float scale = 1.0f / sqrtf((float)d);

    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({B * H, N}, Q.options());

    const int BLOCK = 32;
    dim3 grid(B * H, (N + BLOCK - 1) / BLOCK);
    size_t smem = 2 * BLOCK * d * sizeof(float);

    flash_attention_forward_kernel<BLOCK><<<grid, BLOCK, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), L.data_ptr<float>(),
        N, d, scale);

    return O;
}