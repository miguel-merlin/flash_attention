#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define TILE_WIDTH 16

namespace flash_attention {

// ---------------------------------------------------------------------------
// Forward kernel (stub)
// TODO: Implement the real Flash Attention tiled SRAM algorithm here.
//
// Expected tensor layout: (B, H, N, d) row-major.
// Kernel should tile Q, K, V into SRAM blocks of size Br x d and Bc x d,
// compute softmax incrementally (online normalisation), accumulate into O,
// and write back without materialising the full N×N attention matrix.
// ---------------------------------------------------------------------------
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ query,    // (B*H, N, d)
    const float* __restrict__ key,      // (B*H, N, d)
    const float* __restrict__ value,    // (B*H, N, d)
    float* __restrict__ output,         // (B*H, N, d)
    const int N,
    const int d,
    const float scale)
{
    // TODO: implement tiled flash attention forward
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    (void)idx; (void)query; (void)key; (void)value; (void)output;
    (void)N; (void)d; (void)scale;
}

__global__ void matmul_kernel(
    const float* __restrict__ M,
    const float* __restrict__ N,
    float* __restrict__ P,
    const int M_rows,
    const int M_cols,
    const int N_rows,
    const int N_cols
)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float pval = 0;
    for (unsigned int m = 0; m < (M_cols + TILE_WIDTH -1)/TILE_WIDTH; ++m)
    {
        int mCol = m * TILE_WIDTH + tx;
        int mRow = m * TILE_WIDTH + ty;
        Mds[ty][tx] = (row < M_rows && mCol < M_cols)
                      ? M[row * M_cols + mCol] : 0.0f;
        Nds[ty][tx] = (mRow < N_rows && col < N_cols)
                      ? N[mRow * N_cols + col] : 0.0f;
        __syncthreads();
        for (unsigned int k = 0; k < TILE_WIDTH; ++k)
        {
            pval += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if (row < M_rows && col < N_cols)
        P[row * N_cols + col] = pval;
}

__global__ void softmax_reduction_kernel(
    const float* __restrict__ S,
    float* __restrict__ P,
    const int M,
    const int N
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += expf(S[row * N + i]);
        }
        P[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Backward kernel (stub)
// TODO: Implement the recompute-based Flash Attention backward:
//   - Reload Q, K, V tiles from HBM
//   - Recompute S and P on-the-fly
//   - Accumulate dQ, dK, dV without materialising N×N matrices
// ---------------------------------------------------------------------------
__global__ void flash_attention_backward_kernel(
    const float* __restrict__ grad_output,  // dO  (B*H, N, d)
    const float* __restrict__ query,         // Q
    const float* __restrict__ key,           // K
    const float* __restrict__ value,         // V
    const float* __restrict__ out,           // O  (stored from forward)
    float* __restrict__ grad_query,          // dQ
    float* __restrict__ grad_key,            // dK
    float* __restrict__ grad_value,          // dV
    const int N,
    const int d,
    const float scale)
{
    // TODO: implement tiled flash attention backward
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    (void)idx;
    (void)grad_output; (void)query; (void)key; (void)value; (void)out;
    (void)grad_query; (void)grad_key; (void)grad_value;
    (void)N; (void)d; (void)scale;
}

// ---------------------------------------------------------------------------
// C++ forward container
// ---------------------------------------------------------------------------
at::Tensor flash_attention_cuda(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v)
{
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(),
                "Input tensors must have the same shape");
    TORCH_CHECK(q.dim() == 4,
                "q, k, v must have shape (B, H, N, d)");
    TORCH_CHECK(q.dtype() == at::kFloat,
                "Only float32 is supported");
    TORCH_CHECK(q.device().type() == at::DeviceType::CUDA,
                "Tensors must be on CUDA");
    TORCH_INTERNAL_ASSERT(q.device() == k.device() && k.device() == v.device(),
                          "All tensors must be on the same device");

    at::Tensor q_contig = q.contiguous();
    at::Tensor k_contig = k.contiguous();
    at::Tensor v_contig = v.contiguous();

    at::Tensor result_tensor = at::zeros(q_contig.sizes(), q_contig.options());

    const int64_t B = q_contig.size(0);
    const int64_t H = q_contig.size(1);
    const int64_t N = static_cast<int>(q_contig.size(2));
    const int64_t d = static_cast<int>(q_contig.size(3));
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    const float* q_ptr = q_contig.data_ptr<float>();
    const float* k_ptr = k_contig.data_ptr<float>();
    const float* v_ptr = v_contig.data_ptr<float>();
    float* result_ptr = result_tensor.data_ptr<float>();

    int num_elements = static_cast<int>(B * H * N);
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    float* d_q_times_dk_T = at::empty({B * H, N, N}, q.options()).data_ptr<float>();
    float* d_softmax_times_dv = at::empty({B * H, N, N}, q.options()).data_ptr<float>();

    matmul_kernel<<<blocks, threads, 0, stream>>>(
        q_ptr, k_ptr, d_q_times_dk_T,
        static_cast<int>(B * H), static_cast<int>(N), static_cast<int>(N));
    
    softmax_reduction_kernel<<<blocks, threads, 0, stream>>>(
        d_q_times_dk_T, d_softmax_times_dv,
        static_cast<int>(B * H), static_cast<int>(N));

    return result_tensor;
}

// ---------------------------------------------------------------------------
// C++ backward container
// ---------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor>
flash_attention_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v)
{
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(),
                "Input tensors must have the same shape");
    TORCH_CHECK(q.device().type() == at::DeviceType::CUDA,
                "Tensors must be on CUDA");

    at::Tensor grad_q   = at::zeros_like(q);
    at::Tensor grad_k   = at::zeros_like(k);
    at::Tensor grad_v   = at::zeros_like(v);

    // Placeholder: backward kernel stub is called but produces zeros.
    // Replace with real kernel once implemented.
    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t N = static_cast<int>(q.size(2));
    const int64_t d = static_cast<int>(q.size(3));
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    // dummy forward output (zeros) — real impl would save this from forward
    at::Tensor out_dummy = at::zeros_like(q);

    int num_elements = static_cast<int>(B * H * N);
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    flash_attention_backward_kernel<<<blocks, threads, 0, stream>>>(
        grad_output.contiguous().data_ptr<float>(),
        q.contiguous().data_ptr<float>(),
        k.contiguous().data_ptr<float>(),
        v.contiguous().data_ptr<float>(),
        out_dummy.data_ptr<float>(),
        grad_q.data_ptr<float>(),
        grad_k.data_ptr<float>(),
        grad_v.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(d), scale);

    return std::make_tuple(grad_q, grad_k, grad_v);
}

// ---------------------------------------------------------------------------
// Op registration — CUDA dispatch key
// ---------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(flash_attention, CUDA, m)
{
    m.impl("flash_attention",          flash_attention::flash_attention_cuda);
    m.impl("flash_attention_backward", flash_attention::flash_attention_backward_cuda);
}

} // namespace flash_attention
