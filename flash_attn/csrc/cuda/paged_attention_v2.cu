// paged_attention_v2.cu

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace paged_attention {

__device__ __forceinline__ int64_t q_idx(
    int64_t b, int64_t h, int64_t x,
    int64_t H, int64_t d)
{
    return (b * H + h) * d + x;
}

__device__ __forceinline__ int64_t cache_idx(
    int64_t p, int64_t off, int64_t h, int64_t x,
    int64_t page_size, int64_t H, int64_t d)
{
    return ((p * page_size + off) * H + h) * d + x;
}

__device__ __forceinline__ int64_t score_idx(
    int64_t b, int64_t h, int64_t t,
    int64_t H, int64_t max_seq_len)
{
    return (b * H + h) * max_seq_len + t;
}


// ------------------------------------------------------------
// V2 kernel:
// one CUDA block computes one full output vector out[b, h, :]
//
// Main improvement over V1:
// - scores q·k_t are computed once per token
// - scores/weights are reused for every output dimension x
// ------------------------------------------------------------
__global__ void paged_attention_forward_kernel_v2(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int64_t* __restrict__ page_table,
    const int64_t* __restrict__ seq_lens,
    float* __restrict__ out,
    float* __restrict__ scores,   // temporary: (B, H, max_seq_len)
    int64_t B,
    int64_t H,
    int64_t d,
    int64_t page_size,
    int64_t max_pages,
    int64_t max_seq_len,
    bool /* is_causal */)
{
    int tid = threadIdx.x;

    // One block owns one (b, h)
    int64_t bh = blockIdx.x;
    int64_t h = bh % H;
    int64_t b = bh / H;

    if (b >= B) return;

    int64_t seq_len = seq_lens[b];

    // Dynamic shared memory layout:
    // q_s[d] stores q[b,h,:]
    // red[blockDim.x] is used for reductions
    extern __shared__ float smem[];
    float* q_s  = smem;
    float* red  = smem + d;

    // --------------------------------------------------------
    // Edge case: empty sequence
    // --------------------------------------------------------
    if (seq_len <= 0) {
        for (int64_t x = tid; x < d; x += blockDim.x) {
            out[q_idx(b, h, x, H, d)] = 0.0f;
        }
        return;
    }

    // Safety check. In valid metadata, seq_len <= max_pages * page_size.
    if (seq_len > max_seq_len) {
        return;
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(d));

    // --------------------------------------------------------
    // Step 1: load q[b,h,:] into shared memory
    // --------------------------------------------------------
    for (int64_t x = tid; x < d; x += blockDim.x) {
        q_s[x] = q[q_idx(b, h, x, H, d)];
    }

    __syncthreads();

    // --------------------------------------------------------
    // Step 2: compute raw attention scores once
    //
    // Each thread handles tokens:
    // tid, tid + blockDim.x, tid + 2*blockDim.x, ...
    // --------------------------------------------------------
    float local_max = -INFINITY;

    for (int64_t t = tid; t < seq_len; t += blockDim.x) {
        int64_t logical_page = t / page_size;
        int64_t offset       = t % page_size;

        if (logical_page >= max_pages) {
            return;
        }

        int64_t physical_page = page_table[b * max_pages + logical_page];

        float dot = 0.0f;

        for (int64_t j = 0; j < d; ++j) {
            float q_val = q_s[j];
            float k_val = k_cache[
                cache_idx(
                    physical_page,
                    offset,
                    h,
                    j,
                    page_size,
                    H,
                    d
                )
            ];

            dot += q_val * k_val;
        }

        float s = dot * scale;

        scores[score_idx(b, h, t, H, max_seq_len)] = s;

        if (s > local_max) {
            local_max = s;
        }
    }

    // --------------------------------------------------------
    // Step 3: reduce max_score across the block
    // --------------------------------------------------------
    red[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = red[tid + stride];
            if (other > red[tid]) {
                red[tid] = other;
            }
        }
        __syncthreads();
    }

    float max_score = red[0];

    // --------------------------------------------------------
    // Step 4: compute exp(score - max_score)
    // Store unnormalized weights back into scores[t].
    // --------------------------------------------------------
    float local_sum = 0.0f;

    for (int64_t t = tid; t < seq_len; t += blockDim.x) {
        int64_t sidx = score_idx(b, h, t, H, max_seq_len);

        float w = expf(scores[sidx] - max_score);

        scores[sidx] = w;
        local_sum += w;
    }

    // --------------------------------------------------------
    // Step 5: reduce sum_exp across the block
    // --------------------------------------------------------
    red[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }

    float sum_exp = red[0];

    // --------------------------------------------------------
    // Step 6: compute output vector cooperatively
    //
    // Each thread owns output dimensions:
    // tid, tid + blockDim.x, tid + 2*blockDim.x, ...
    // Since d is usually 64 or 128, only first d threads
    // will actively compute output dimensions.
    // --------------------------------------------------------
    for (int64_t x = tid; x < d; x += blockDim.x) {
        float acc = 0.0f;

        for (int64_t t = 0; t < seq_len; ++t) {
            int64_t logical_page = t / page_size;
            int64_t offset       = t % page_size;
            int64_t physical_page = page_table[b * max_pages + logical_page];

            float w = scores[score_idx(b, h, t, H, max_seq_len)] / sum_exp;

            float v_val = v_cache[
                cache_idx(
                    physical_page,
                    offset,
                    h,
                    x,
                    page_size,
                    H,
                    d
                )
            ];

            acc += w * v_val;
        }

        out[q_idx(b, h, x, H, d)] = acc;
    }
}


at::Tensor paged_attention_cuda_v2(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& page_table,
    const at::Tensor& seq_lens,
    bool is_causal)
{
    TORCH_CHECK(q.device().type() == at::DeviceType::CUDA, "q must be CUDA");
    TORCH_CHECK(k_cache.device().type() == at::DeviceType::CUDA, "k_cache must be CUDA");
    TORCH_CHECK(v_cache.device().type() == at::DeviceType::CUDA, "v_cache must be CUDA");
    TORCH_CHECK(page_table.device().type() == at::DeviceType::CUDA, "page_table must be CUDA");
    TORCH_CHECK(seq_lens.device().type() == at::DeviceType::CUDA, "seq_lens must be CUDA");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "q must be float32");
    TORCH_CHECK(k_cache.scalar_type() == at::kFloat, "k_cache must be float32");
    TORCH_CHECK(v_cache.scalar_type() == at::kFloat, "v_cache must be float32");

    TORCH_CHECK(q.dim() == 3, "q must have shape (B, H, d)");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must have shape (P, page_size, H, d)");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must have shape (P, page_size, H, d)");
    TORCH_CHECK(page_table.dim() == 2, "page_table must have shape (B, max_pages)");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape (B,)");

    auto q_contig = q.contiguous();
    auto k_contig = k_cache.contiguous();
    auto v_contig = v_cache.contiguous();

    auto pt_long = page_table.contiguous().to(at::kLong);
    auto sl_long = seq_lens.contiguous().to(at::kLong);

    int64_t B = q_contig.size(0);
    int64_t H = q_contig.size(1);
    int64_t d = q_contig.size(2);

    int64_t P = k_contig.size(0);
    int64_t page_size = k_contig.size(1);
    int64_t Hk = k_contig.size(2);
    int64_t dk = k_contig.size(3);

    TORCH_CHECK(Hk == H, "k_cache H must match q H");
    TORCH_CHECK(dk == d, "k_cache d must match q d");
    TORCH_CHECK(v_contig.sizes() == k_contig.sizes(), "v_cache must match k_cache shape");

    TORCH_CHECK(pt_long.size(0) == B, "page_table first dim must be B");

    int64_t max_pages = pt_long.size(1);
    int64_t max_seq_len = max_pages * page_size;

    auto out = at::empty_like(q_contig);

    // Temporary score/weight buffer:
    // first stores raw scores, then stores exp(score - max)
    auto scores = at::empty({B, H, max_seq_len}, q_contig.options());

    constexpr int threads = 256;
    dim3 block(threads);
    dim3 grid(B * H);

    // shared memory:
    // q_s[d] + red[threads]
    size_t shared_mem_bytes =
        static_cast<size_t>(d + threads) * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    paged_attention_forward_kernel_v2<<<
        grid,
        block,
        shared_mem_bytes,
        stream
    >>>(
        q_contig.data_ptr<float>(),
        k_contig.data_ptr<float>(),
        v_contig.data_ptr<float>(),
        pt_long.data_ptr<int64_t>(),
        sl_long.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        scores.data_ptr<float>(),
        B,
        H,
        d,
        page_size,
        max_pages,
        max_seq_len,
        is_causal
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

} // namespace paged_attention