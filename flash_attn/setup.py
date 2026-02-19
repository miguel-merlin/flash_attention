import os
import platform
import shutil
from setuptools import setup


def _fail(msg: str) -> None:
    raise RuntimeError(f"\n[flash_attn setup error] {msg}\n")


def _preflight_checks() -> None:
    if platform.system() != "Linux":
        _fail(
            "This package builds a CUDA extension and is supported only on Linux with an NVIDIA GPU.\n"
            f"Detected platform: {platform.system()} ({platform.machine()})."
        )

    try:
        import torch
    except ModuleNotFoundError:
        _fail(
            "PyTorch is required at build time but is not importable.\n"
            "Install torch first, then run editable install without build isolation:\n"
            "  pip install torch\n"
            "  pip install -e flash_attn --no-build-isolation"
        )
    except Exception as exc:
        _fail(f"Failed to import torch during build setup: {exc}")

    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        _fail(
            "CUDA_HOME is not set.\n"
            "Set CUDA_HOME to your CUDA toolkit root (example: /usr/local/cuda) and retry."
        )

    nvcc = shutil.which("nvcc")
    if not nvcc:
        _fail(
            "CUDA toolkit compiler 'nvcc' was not found in PATH.\n"
            "Install the CUDA toolkit and ensure 'nvcc' is available."
        )


_preflight_checks()

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash_attn_cuda",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_cuda",
            sources=["csrc/flash_attn.cpp", "csrc/flash_attn.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-arch=sm_80"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
