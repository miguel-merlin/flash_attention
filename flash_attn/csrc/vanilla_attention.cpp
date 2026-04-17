#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <cmath>

namespace vanilla_attention
{
    at::Tensor vanilla_attention_cpu(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, bool is_causal)
    {
        TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(), "q, k, v must have the same shape");
        TORCH_CHECK(q.dim() == 4, "q, k, v must have shape (B, H, N, d)");
        TORCH_CHECK(q.dtype() == at::kFloat, "Only float32 is supported for q, k, v");
        TORCH_CHECK(q.dtype() == k.dtype() && k.dtype() == v.dtype(), "q, k, v must have the same dtype");
        TORCH_INTERNAL_ASSERT(q.device().type() == at::DeviceType::CPU, "q must be on CPU");
        TORCH_INTERNAL_ASSERT(k.device().type() == at::DeviceType::CPU, "k must be on CPU");
        TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CPU, "v must be on CPU");

        at::Tensor q_contig = q.contiguous();
        at::Tensor k_contig = k.contiguous();
        at::Tensor v_contig = v.contiguous();

        const int64_t B = q_contig.size(0);
        const int64_t H = q_contig.size(1);
        const int64_t N = q_contig.size(2);
        const int64_t d = q_contig.size(3);

        const float scale = 1.0f / std::sqrt(static_cast<float>(d));

        at::Tensor output = torch::zeros(q_contig.sizes(), q_contig.options());

        const float* q_ptr = q_contig.data_ptr<float>();
        const float* k_ptr = k_contig.data_ptr<float>();
        const float* v_ptr = v_contig.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        auto idx4 = [H, N, d](int64_t b, int64_t h, int64_t n, int64_t x) {
            return ((b * H + h) * N + n) * d + x;
        };

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t i = 0; i < N; ++i) {
                    std::vector<float> scores(N, 0.0f);
                    for (int64_t j = 0; j < N; ++j) {
                        if (is_causal && j > i) {
                            scores[j] = -INFINITY;
                        } else {
                            float dot = 0.0f;
                            for (int64_t x = 0; x < d; ++x) {
                                dot += q_ptr[idx4(b, h, i, x)] * k_ptr[idx4(b, h, j, x)];
                            }
                            scores[j] = dot * scale;
                        }
                    }

                    float row_max = scores[0];
                    for (int64_t j = 1; j < N; ++j) {
                        if (scores[j] > row_max) row_max = scores[j];
                    }

                    std::vector<float> probs(N, 0.0f);
                    float exp_sum = 0.0f;
                    for (int64_t j = 0; j < N; ++j) {
                        probs[j] = std::exp(scores[j] - row_max);
                        exp_sum += probs[j];
                    }
                    for (int64_t j = 0; j < N; ++j) {
                        probs[j] /= exp_sum;
                    }

                    for (int64_t x = 0; x < d; ++x) {
                        float acc = 0.0f;
                        for (int64_t j = 0; j < N; ++j) {
                            acc += probs[j] * v_ptr[idx4(b, h, j, x)];
                        }
                        output_ptr[idx4(b, h, i, x)] = acc;
                    }
                }
            }
        }

        return output;
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor>
    vanilla_attention_backward_cpu(
        const at::Tensor &grad_output,
        const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        bool is_causal)
    {
        TORCH_CHECK(false,
            "vanilla_attention_backward_cpu: C++ backward not yet implemented. "
            "Gradients are computed in Python via ops.py.");
        // Unreachable
        return std::make_tuple(
            at::zeros_like(q), at::zeros_like(k), at::zeros_like(v));
    }

    // -----------------------------------------------------------------------
    // Op registration
    // -----------------------------------------------------------------------
    TORCH_LIBRARY(vanilla_attention, m)
    {
        // Forward op: (Tensor q, Tensor k, Tensor v, bool is_causal) -> Tensor
        m.def("vanilla_attention(Tensor q, Tensor k, Tensor v, bool is_causal=False) -> Tensor");

        // Backward op: (Tensor grad, Tensor q, Tensor k, Tensor v, bool is_causal) -> (Tensor, Tensor, Tensor)
        m.def("vanilla_attention_backward(Tensor grad_output, Tensor q, Tensor k, Tensor v, bool is_causal=False) -> (Tensor, Tensor, Tensor)");
    }

    TORCH_LIBRARY_IMPL(vanilla_attention, CPU, m)
    {
        m.impl("vanilla_attention", vanilla_attention::vanilla_attention_cpu);
        m.impl("vanilla_attention_backward", vanilla_attention::vanilla_attention_backward_cpu);
    }

#ifdef WITH_CUDA
    extern at::Tensor vanilla_attention_cuda(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, bool is_causal);
    extern std::tuple<at::Tensor, at::Tensor, at::Tensor> vanilla_attention_backward_cuda(
        const at::Tensor &grad_output, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, bool is_causal);

    TORCH_LIBRARY_IMPL(vanilla_attention, CUDA, m)
    {
        m.impl("vanilla_attention", vanilla_attention_cuda);
        m.impl("vanilla_attention_backward", vanilla_attention_backward_cuda);
    }
#endif

} // namespace vanilla_attention

int vanilla_attention_cpp_force_link() { return 0; }
