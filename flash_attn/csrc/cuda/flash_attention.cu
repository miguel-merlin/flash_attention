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
// Forward kernel
// ---------------------------------------------------------------------------
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ query,    // (B*H, N, d)
    const float* __restrict__ key,      // (B*H, N, d)
    const float* __restrict__ value,    // (B*H, N, d)
    float* __restrict__ output,         // (B*H, N, d)
    const int N,
    const int d,
    const float scale,
    const int Br,
    const int Bc)
{
    int bx = blockIdx.y; 
    int q_block_idx = blockIdx.x;
    
    int tx = threadIdx.x;
    int num_threads = blockDim.x;

    int bh_offset = bx * N * d;
    
    extern __shared__ float sram[];
    float* Qi = sram;                     // Br x d
    float* Kj = &sram[Br * d];            // Bc x d
    float* Vj = &sram[(Br + Bc) * d];     // Bc x d
    float* S  = &sram[(Br + Bc * 2) * d]; // Br x Bc
    float* O_i = &sram[(Br + Bc * 2) * d + Br * Bc]; // Br x d
    
    float* m_i_arr = &sram[(Br + Bc * 2) * d + Br * Bc + Br * d];
    float* l_i_arr = &m_i_arr[Br];
    float* exp_diff_arr = &l_i_arr[Br];
    
    int start_q = q_block_idx * Br;
    
    // Load Q block
    for (int j = tx; j < Br * d; j += num_threads) {
        int row = j / d;
        int col = j % d;
        if (start_q + row < N) {
            Qi[row * d + col] = query[bh_offset + (start_q + row) * d + col];
        } else {
            Qi[row * d + col] = 0.0f;
        }
    }
    
    // Initialize O_i
    for (int j = tx; j < Br * d; j += num_threads) {
        O_i[j] = 0.0f;
    }
    
    // Initialize running max and denominator
    for (int j = tx; j < Br; j += num_threads) {
        m_i_arr[j] = -INFINITY;
        l_i_arr[j] = 0.0f;
    }
    
    __syncthreads();
    
    int Tc = (N + Bc - 1) / Bc;
    
    for (int c = 0; c < Tc; c++) {
        int start_k = c * Bc;
        
        // Load K and V blocks
        for (int j = tx; j < Bc * d; j += num_threads) {
            int row = j / d;
            int col = j % d;
            if (start_k + row < N) {
                Kj[row * d + col] = key[bh_offset + (start_k + row) * d + col];
                Vj[row * d + col] = value[bh_offset + (start_k + row) * d + col];
            } else {
                Kj[row * d + col] = 0.0f;
                Vj[row * d + col] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute S = Q * K^T
        for (int j = tx; j < Br * Bc; j += num_threads) {
            int row = j / Bc;
            int col = j % Bc;
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Qi[row * d + k] * Kj[col * d + k];
            }
            S[row * Bc + col] = sum * scale;
        }
        __syncthreads();
        
        // Compute max and exponents
        if (tx < Br) {
            int row = tx;
            float m_ij = -INFINITY;
            for (int k = 0; k < Bc; k++) {
                if (start_k + k < N) {
                    m_ij = max(m_ij, S[row * Bc + k]);
                }
            }
            float m_i_old = m_i_arr[row];
            float m_i_new = max(m_i_old, m_ij);
            exp_diff_arr[row] = expf(m_i_old - m_i_new);
            m_i_arr[row] = m_i_new;
            
            float l_ij = 0.0f;
            for (int k = 0; k < Bc; k++) {
                if (start_k + k < N) {
                    float val = expf(S[row * Bc + k] - m_i_new);
                    S[row * Bc + k] = val; // Store P_ij
                    l_ij += val;
                } else {
                    S[row * Bc + k] = 0.0f;
                }
            }
            l_i_arr[row] = exp_diff_arr[row] * l_i_arr[row] + l_ij;
        }
        __syncthreads();
        
        // Update O_i
        for (int j = tx; j < Br * d; j += num_threads) {
            int row = j / d;
            int col = j % d;
            float pv = 0.0f;
            for (int k = 0; k < Bc; k++) {
                pv += S[row * Bc + k] * Vj[k * d + col];
            }
            O_i[row * d + col] = exp_diff_arr[row] * O_i[row * d + col] + pv;
        }
        __syncthreads();
    }
    
    // Write out
    for (int j = tx; j < Br * d; j += num_threads) {
        int row = j / d;
        int col = j % d;
        if (start_q + row < N) {
            output[bh_offset + (start_q + row) * d + col] = O_i[row * d + col] / l_i_arr[row];
        }
    }
}

// ---------------------------------------------------------------------------
// Backward kernel (stub)
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
    // Placeholder for backward
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
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(q.dim() == 4, "q, k, v must have shape (B, H, N, d)");
    TORCH_CHECK(q.dtype() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(q.device().type() == at::DeviceType::CUDA, "Tensors must be on CUDA");

    at::Tensor q_contig = q.contiguous();
    at::Tensor k_contig = k.contiguous();
    at::Tensor v_contig = v.contiguous();

    at::Tensor result_tensor = at::zeros(q_contig.sizes(), q_contig.options());

    const int64_t B = q_contig.size(0);
    const int64_t H = q_contig.size(1);
    const int64_t N = static_cast<int>(q_contig.size(2));
    const int64_t d = static_cast<int>(q_contig.size(3));
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    int Br = (d <= 64) ? 32 : 16;
    int Bc = (d <= 64) ? 32 : 16;
    
    int shared_mem_size = (2 * Br * d + 2 * Bc * d + Br * Bc + 3 * Br) * sizeof(float);
    
    int num_blocks_x = (N + Br - 1) / Br;
    int num_blocks_y = B * H;
    
    dim3 grid(num_blocks_x, num_blocks_y);
    dim3 block(256);

    const float* q_ptr = q_contig.data_ptr<float>();
    const float* k_ptr = k_contig.data_ptr<float>();
    const float* v_ptr = v_contig.data_ptr<float>();
    float* result_ptr = result_tensor.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    flash_attention_forward_kernel<<<grid, block, shared_mem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, result_ptr,
        static_cast<int>(N), static_cast<int>(d), scale, Br, Bc
    );

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
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(q.device().type() == at::DeviceType::CUDA, "Tensors must be on CUDA");

    at::Tensor grad_q = at::zeros_like(q);
    at::Tensor grad_k = at::zeros_like(k);
    at::Tensor grad_v = at::zeros_like(v);

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t N = static_cast<int>(q.size(2));
    const int64_t d = static_cast<int>(q.size(3));
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

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

} // namespace flash_attention



