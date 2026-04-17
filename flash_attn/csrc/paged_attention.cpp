// paged_attention.cpp
#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <algorithm>

#include <vector>
#include <tuple>
#include <cmath>
#include <cstdint>

namespace paged_attention {

at::Tensor paged_attention_cpu(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& page_table,
    const at::Tensor& seq_lens,
    bool /* is_causal */)
{
    // is_causal: reserved for API parity with flash_attention(q,k,v,is_causal).
    // Decode-style paged attention only ever reads past KV (0 .. seq_len-1), so
    // masking matches full causal attention over the prefix; no extra mask here.
    TORCH_CHECK(q.dim() == 3, "q must have shape (B, H, d)");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must have shape (P, page_size, H, d)");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must have shape (P, page_size, H, d)");
    TORCH_CHECK(page_table.dim() == 2, "page_table must have shape (B, max_pages)");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape (B,)");
    TORCH_CHECK(q.scalar_type() == at::kFloat, "Only float32 is supported for q");
    TORCH_CHECK(k_cache.scalar_type() == at::kFloat && v_cache.scalar_type() == at::kFloat,
                "Only float32 is supported for k_cache and v_cache");
    TORCH_CHECK(
        page_table.scalar_type() == at::kInt || page_table.scalar_type() == at::kLong,
        "page_table must be int32 or int64");
    TORCH_CHECK(
        seq_lens.scalar_type() == at::kInt || seq_lens.scalar_type() == at::kLong,
        "seq_lens must be int32 or int64");
    TORCH_INTERNAL_ASSERT(q.device().type() == at::DeviceType::CPU, "q must be on CPU");
    TORCH_INTERNAL_ASSERT(k_cache.device().type() == at::DeviceType::CPU, "k_cache must be on CPU");
    TORCH_INTERNAL_ASSERT(v_cache.device().type() == at::DeviceType::CPU, "v_cache must be on CPU");
    TORCH_INTERNAL_ASSERT(page_table.device().type() == at::DeviceType::CPU,
                          "page_table must be on CPU");
    TORCH_INTERNAL_ASSERT(seq_lens.device().type() == at::DeviceType::CPU,
                          "seq_lens must be on CPU");

    auto q_contig = q.contiguous();
    auto k_contig = k_cache.contiguous();
    auto v_contig = v_cache.contiguous();
    auto pt_contig = page_table.contiguous();
    auto sl_contig = seq_lens.contiguous();

    const int64_t B = q_contig.size(0);
    const int64_t H = q_contig.size(1);
    const int64_t d = q_contig.size(2);

    const int64_t P = k_contig.size(0);
    const int64_t page_size = k_contig.size(1);

    TORCH_CHECK(k_contig.size(2) == H, "k_cache H mismatch");
    TORCH_CHECK(v_contig.size(2) == H, "v_cache H mismatch");
    TORCH_CHECK(k_contig.size(3) == d, "k_cache d mismatch");
    TORCH_CHECK(v_contig.size(3) == d, "v_cache d mismatch");
    TORCH_CHECK(v_contig.size(0) == P, "v_cache P mismatch");
    TORCH_CHECK(v_contig.size(1) == page_size, "v_cache page_size mismatch");
    TORCH_CHECK(pt_contig.size(0) == B, "page_table batch mismatch");
    TORCH_CHECK(sl_contig.size(0) == B, "seq_lens batch mismatch");

    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    auto out = torch::zeros({B, H, d}, q_contig.options());

    const float* q_ptr = q_contig.data_ptr<float>();
    const float* k_ptr = k_contig.data_ptr<float>();
    const float* v_ptr = v_contig.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    auto q_idx = [H, d](int64_t b, int64_t h, int64_t x) {
        return (b * H + h) * d + x;
    };

    auto cache_idx = [page_size, H, d](int64_t p, int64_t off, int64_t h, int64_t x) {
        return ((p * page_size + off) * H + h) * d + x;
    };

    const int64_t max_pages = pt_contig.size(1);

    for (int64_t b = 0; b < B; ++b) {
        int64_t seq_len =
            (sl_contig.scalar_type() == at::kInt)
                ? static_cast<int64_t>(sl_contig.data_ptr<int32_t>()[b])
                : static_cast<int64_t>(sl_contig.data_ptr<int64_t>()[b]);

        if (seq_len <= 0) {
            continue;
        }

        TORCH_CHECK(
            seq_len <= max_pages * page_size,
            "seq_lens[", b, "] implies more tokens than max_pages * page_size");

        for (int64_t h = 0; h < H; ++h) {
            std::vector<float> scores(seq_len);

            for (int64_t t = 0; t < seq_len; ++t) {
                int64_t logical_page = t / page_size;
                int64_t offset = t % page_size;

                int64_t physical_page =
                    (pt_contig.scalar_type() == at::kInt)
                        ? static_cast<int64_t>(
                              pt_contig.data_ptr<int32_t>()[b * max_pages + logical_page])
                        : pt_contig.data_ptr<int64_t>()[b * max_pages + logical_page];

                float dot = 0.0f;
                for (int64_t x = 0; x < d; ++x) {
                    dot += q_ptr[q_idx(b, h, x)] * k_ptr[cache_idx(physical_page, offset, h, x)];
                }

                scores[t] = dot * scale;
            }

            float max_val = *std::max_element(scores.begin(), scores.end());

            std::vector<float> probs(seq_len);
            float sum = 0.0f;

            for (int64_t t = 0; t < seq_len; ++t) {
                probs[t] = std::exp(scores[t] - max_val);
                sum += probs[t];
            }

            for (int64_t x = 0; x < d; ++x) {
                float acc = 0.0f;

                for (int64_t t = 0; t < seq_len; ++t) {
                    int64_t logical_page = t / page_size;
                    int64_t offset = t % page_size;

                    int64_t physical_page =
                        (pt_contig.scalar_type() == at::kInt)
                            ? static_cast<int64_t>(pt_contig.data_ptr<int32_t>()
                                                       [b * max_pages + logical_page])
                            : pt_contig.data_ptr<int64_t>()[b * max_pages + logical_page];

                    float p = probs[t] / sum;

                    acc += p * v_ptr[cache_idx(physical_page, offset, h, x)];
                }

                out_ptr[q_idx(b, h, x)] = acc;
            }
        }
    }

    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
paged_attention_backward_cpu(
    const at::Tensor&,
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor&,
    const at::Tensor&,
    bool /* is_causal */)
{
    TORCH_CHECK(false,
                "paged_attention_backward_cpu: C++ backward not yet implemented. "
                "Gradients are computed in Python via ops.py.");
    return std::make_tuple(
        at::zeros_like(q), at::zeros_like(k_cache), at::zeros_like(v_cache));
}

TORCH_LIBRARY_FRAGMENT(flash_attention, m)
{
    m.def(
        "paged_attention(Tensor q, Tensor k_cache, Tensor v_cache, Tensor page_table, Tensor "
        "seq_lens, bool is_causal=False) -> Tensor");
    m.def(
        "paged_attention_backward(Tensor grad_output, Tensor q, Tensor k_cache, Tensor v_cache, "
        "Tensor page_table, Tensor seq_lens, bool is_causal=False) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flash_attention, CPU, m)
{
    m.impl("paged_attention", paged_attention_cpu);
    m.impl("paged_attention_backward", paged_attention_backward_cpu);
}

#ifdef WITH_CUDA
extern at::Tensor paged_attention_cuda(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& page_table,
    const at::Tensor& seq_lens,
    bool is_causal);

extern std::tuple<at::Tensor, at::Tensor, at::Tensor> paged_attention_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& page_table,
    const at::Tensor& seq_lens,
    bool is_causal);

TORCH_LIBRARY_IMPL(flash_attention, CUDA, m)
{
    m.impl("paged_attention", paged_attention_cuda);
    m.impl("paged_attention_backward", paged_attention_backward_cuda);
}
#endif

} // namespace paged_attention
