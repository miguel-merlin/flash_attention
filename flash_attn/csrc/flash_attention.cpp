#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>
extern "C" {
    /*
        Creates a dummye empty _C module that can be imported from Python.
        The import from Python will load the .so consisting of this file
        in this extension, so that the TORCH_LIBRARY statuc inits below are run
    */
   PyObject* PyInit__C(void) 
   {
        static struct PyModuleDef module_def = 
        {
            PyModuleDef_HEAD_INIT,
            "_C",
            NULL,
            -1,
            NULL
        };
        return PyModule_Create(&module_def);
   }
}

namespace flash_attention 
{
at::Tensor flash_attention_cpu(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v)
{
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(), "q, k, v must have the same shape");
    TORCH_CHECK(q.dtype() == at::kFloat, "Only float32 is supported for q, k, v");
    TORCH_CHECK(q.dtype() == k.dtype() && k.dtype() == v.dtype(), "q, k, v must have the same dtype");
    TORCH_INTERNAL_ASSERT(q.device().type() == at::DeviceType::CPU, "q must be on CPU");
    TORCH_INTERNAL_ASSERT(k.device().type() == at::DeviceType::CPU, "k must be on CPU");
    TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CPU, "v must be on CPU");
    at::Tensor q_contig = q.contiguous();
    at::Tensor k_contig = k.contiguous();
    at::Tensor v_contig = v.contiguous();
    at::Tensor output = torch::empty(q_contig.sizes(), q_contig.options());
    const float* q_ptr = q_contig.data_ptr<float>();
    const float* k_ptr = k_contig.data_ptr<float>();
    const float* v_ptr = v_contig.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    for (int64_t i = 0; i < q_contig.numel(); ++i) {
        output_ptr[i] = q_ptr[i] + k_ptr[i] + v_ptr[i];
    }
    return output;
}
TORCH_LIBRARY(flash_attention, m) 
{
    m.def("flash_attention(Tensor q, Tensor k, Tensor v -> Tensor)");
}

TORCH_LIBRARY_IMPL(flash_attention, CPU, m) 
{
    m.impl("flash_attention", flash_attention::flash_attention_cpu);
}
}