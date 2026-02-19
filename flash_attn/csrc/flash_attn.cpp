#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")

torch::Tensor flash_attention_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attention_forward", &flash_attention_forward,
          "Flash Attention forward (CUDA)");
}