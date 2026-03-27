import os
import glob
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME
)
library_name = "flash_attention"
if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")
        
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    extra_link_args = [f"-Wl,-rpath,{torch_lib_dir}"]
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=" + ("0x03060000" if py_limited_api else "0"),
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ]
    }
    if debug_mode:
        extra_compile_args["cxx"].extend(["-g"])
        extra_compile_args["nvcc"].extend(["-g"])
        extra_link_args.extend(['-O0', '-g'])
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_cuda_dir = os.path.join(this_dir, "flash_attn", "csrc")
    sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cpp")))
    
    extensions_cuda_dir = os.path.join(extensions_cuda_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))
    
    if use_cuda:
        sources += cuda_sources
    
    ext_modules = [
        extension(
            "flash_attn._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api
        )
    ]
    return ext_modules

setup(
    name=library_name,
    version="0.1.0",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch>=2.0.0"],
    description="Flash Attention implementation in PyTorch",
    long_description=open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
    ).read(),
    long_description_content_type="text/markdown",
    url="https://github.com/miguel-merlin/flash_attention",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
    