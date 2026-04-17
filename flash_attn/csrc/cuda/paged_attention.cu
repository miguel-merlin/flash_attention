// paged_attention.cu
// Lean includes only: torch/all.h pulls pybind11 through nvcc and breaks with Py_LIMITED_API
// (see https://github.com/pytorch/pytorch/issues/69460 — use ATen in .cu host code).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <tuple>

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


// Naive kernel:
// one thread computes exactly one output scalar out[b, h, x]
__global__ void paged_attention_forward_kernel_naive(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int64_t* __restrict__ page_table,
    const int64_t* __restrict__ seq_lens,
    float* __restrict__ out,
    int64_t B,
    int64_t H,
    int64_t d,
    int64_t page_size,
    int64_t max_pages,
    bool /* is_causal */)
{
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = B * H * d;
    if (flat >= total) return;

    // Decode flat -> (b, h, x)
    int64_t x = flat % d;
    int64_t tmp = flat / d;
    int64_t h = tmp % H;
    int64_t b = tmp / H;

    int64_t seq_len = seq_lens[b];
    if (seq_len <= 0) {
        out[q_idx(b, h, x, H, d)] = 0.0f;
        return;
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(d));

    // -----------------------------------------
    // Pass 1: find max score for stable softmax
    // -----------------------------------------
    float max_score = -INFINITY;

    for (int64_t t = 0; t < seq_len; ++t) {
        int64_t logical_page = t / page_size;
        int64_t offset       = t % page_size;

        // safety: seq_len should imply enough pages exist
        if (logical_page >= max_pages) {
            return; // invalid metadata; avoid out-of-bounds
        }

        int64_t physical_page = page_table[b * max_pages + logical_page];

        float dot = 0.0f;
        for (int64_t j = 0; j < d; ++j) {
            dot += q[q_idx(b, h, j, H, d)] *
                   k_cache[cache_idx(physical_page, offset, h, j, page_size, H, d)];
        }

        float score = dot * scale;
        if (score > max_score) max_score = score;
    }

    // ------------------------------------------------
    // Pass 2: compute sum exp(score - max) and output
    // ------------------------------------------------
    float sum_exp = 0.0f;
    float weighted_sum = 0.0f;

    for (int64_t t = 0; t < seq_len; ++t) {
        int64_t logical_page = t / page_size;
        int64_t offset       = t % page_size;
        int64_t physical_page = page_table[b * max_pages + logical_page];

        float dot = 0.0f;
        for (int64_t j = 0; j < d; ++j) {
            dot += q[q_idx(b, h, j, H, d)] *
                   k_cache[cache_idx(physical_page, offset, h, j, page_size, H, d)];
        }

        float score = dot * scale;
        float w = expf(score - max_score);

        sum_exp += w;
        weighted_sum += w * v_cache[cache_idx(physical_page, offset, h, x, page_size, H, d)];
    }

    out[q_idx(b, h, x, H, d)] = weighted_sum / sum_exp;
}


// -------------------------------------
// CUDA forward wrapper
// -------------------------------------
at::Tensor paged_attention_cuda(
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

    // For the first naive version, force metadata to int64 for simplicity.
    auto pt_long = page_table.contiguous().to(at::kLong);
    auto sl_long = seq_lens.contiguous().to(at::kLong);

    const int64_t B = q_contig.size(0);
    const int64_t H = q_contig.size(1);
    const int64_t d = q_contig.size(2);

    TORCH_CHECK(k_contig.size(2) == H, "k_cache H mismatch");
    TORCH_CHECK(v_contig.size(2) == H, "v_cache H mismatch");
    TORCH_CHECK(k_contig.size(3) == d, "k_cache d mismatch");
    TORCH_CHECK(v_contig.size(3) == d, "v_cache d mismatch");
    TORCH_CHECK(pt_long.size(0) == B, "page_table batch mismatch");
    TORCH_CHECK(sl_long.size(0) == B, "seq_lens batch mismatch");

    const int64_t P = k_contig.size(0);
    const int64_t page_size = k_contig.size(1);
    const int64_t max_pages = pt_long.size(1);

    (void)P; // unused in this first kernel except for shape validation
    TORCH_CHECK(v_contig.size(0) == P, "v_cache P mismatch");
    TORCH_CHECK(v_contig.size(1) == page_size, "v_cache page_size mismatch");

    auto out = at::zeros({B, H, d}, q_contig.options());

    const int threads = 256;
    const int64_t total = B * H * d;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    paged_attention_forward_kernel_naive<<<blocks, threads, 0, stream>>>(
        q_contig.data_ptr<float>(),
        k_contig.data_ptr<float>(),
        v_contig.data_ptr<float>(),
        pt_long.data_ptr<int64_t>(),
        sl_long.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        B, H, d, page_size, max_pages,
        is_causal);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}


// -------------------------------------
// CUDA backward stub
// -------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor>
paged_attention_backward_cuda(
    const at::Tensor& /* grad_output */,
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& /* page_table */,
    const at::Tensor& /* seq_lens */,
    bool /* is_causal */)
{
    TORCH_CHECK(false, "paged_attention_backward CUDA not implemented");
    return std::make_tuple(
        at::zeros_like(q), at::zeros_like(k_cache), at::zeros_like(v_cache));
}

} // namespace paged_attention

int paged_attention_cuda_force_link()
{
    return 0;
}
