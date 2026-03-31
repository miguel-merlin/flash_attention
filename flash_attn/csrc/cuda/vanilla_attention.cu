#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

namespace vanilla_attention {

// Kernel: S = Q * K^T * scale
__global__ void bmm_qk_kernel(
    const float* __restrict__ Q,    // (B*H, N, d)
    const float* __restrict__ K,    // (B*H, N, d)
    float* __restrict__ S,          // (B*H, N, N)
    const int N,
    const int d,
    const float scale)
{
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        int offset_q = batch_idx * N * d + row * d;
        int offset_k = batch_idx * N * d + col * d;

        for (int i = 0; i < d; ++i) {
            sum += Q[offset_q + i] * K[offset_k + i];
        }
        S[batch_idx * N * N + row * N + col] = sum * scale;
    }
}

// Kernel: P = softmax(S, dim=-1)
__global__ void softmax_kernel(
    float* __restrict__ SP,         // (B*H, N, N) - in/out
    const int N)
{
    int batch_idx = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        int offset = batch_idx * N * N + row * N;
        
        // Find max
        float max_val = -INFINITY;
        for (int i = 0; i < N; ++i) {
            max_val = max(max_val, SP[offset + i]);
        }
        
        // Exp sum
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            float e = expf(SP[offset + i] - max_val);
            SP[offset + i] = e;
            sum += e;
        }
        
        // Normalize
        for (int i = 0; i < N; ++i) {
            SP[offset + i] /= sum;
        }
    }
}

// Kernel: O = P * V
__global__ void bmm_pv_kernel(
    const float* __restrict__ P,    // (B*H, N, N)
    const float* __restrict__ V,    // (B*H, N, d)
    float* __restrict__ O,          // (B*H, N, d)
    const int N,
    const int d)
{
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < d) {
        float sum = 0.0f;
        int offset_p = batch_idx * N * N + row * N;
        int offset_v = batch_idx * N * d;

        for (int i = 0; i < N; ++i) {
            sum += P[offset_p + i] * V[offset_v + i * d + col];
        }
        O[batch_idx * N * d + row * d + col] = sum;
    }
}

at::Tensor vanilla_attention_cuda(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v)
{
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(q.dim() == 4, "q, k, v must have shape (B, H, N, d)");
    TORCH_CHECK(q.dtype() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(q.device().type() == at::DeviceType::CUDA, "Tensors must be on CUDA");

    at::Tensor q_contig = q.contiguous();
    at::Tensor k_contig = k.contiguous();
    at::Tensor v_contig = v.contiguous();

    const int64_t B = q_contig.size(0);
    const int64_t H = q_contig.size(1);
    const int64_t N = static_cast<int>(q_contig.size(2));
    const int64_t d = static_cast<int>(q_contig.size(3));
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    at::Tensor S = at::empty({B * H, N, N}, q_contig.options());
    at::Tensor output = at::empty({B, H, N, d}, q_contig.options());

    const float* q_ptr = q_contig.data_ptr<float>();
    const float* k_ptr = k_contig.data_ptr<float>();
    const float* v_ptr = v_contig.data_ptr<float>();
    float* S_ptr = S.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 1. S = QK^T
    dim3 block_qk(16, 16);
    dim3 grid_qk((N + 15) / 16, (N + 15) / 16, B * H);
    bmm_qk_kernel<<<grid_qk, block_qk, 0, stream>>>(
        q_ptr, k_ptr, S_ptr, N, d, scale
    );

    // 2. Softmax (Row-wise max and reduce)
    dim3 block_sm(256);
    dim3 grid_sm((N + 255) / 256, B * H);
    softmax_kernel<<<grid_sm, block_sm, 0, stream>>>(
        S_ptr, N
    );

    // 3. O = PV
    dim3 block_pv(16, 16);
    dim3 grid_pv((d + 15) / 16, (N + 15) / 16, B * H);
    bmm_pv_kernel<<<grid_pv, block_pv, 0, stream>>>(
        S_ptr, v_ptr, out_ptr, N, d
    );

    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
vanilla_attention_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v)
{
    // Dummy return, backward is handled in Python level
    return std::make_tuple(at::zeros_like(q), at::zeros_like(k), at::zeros_like(v));
}

} // namespace vanilla_attention

int vanilla_attention_cuda_force_link() { return 0; }
