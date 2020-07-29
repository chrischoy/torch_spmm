import sys
from sys import argv, platform
import torch.cuda
import os
import subprocess
from setuptools import setup
import unittest
from pathlib import Path

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


def run_command(*args):
    subprocess.check_call(args)


def _argparse(pattern, argv, is_flag=True):
    if is_flag:
        found = pattern in argv
        if found:
            argv.remove(pattern)
        return found, argv
    else:
        arr = [arg for arg in argv if pattern in arg]
        if len(arr) == 0:  # not found
            return False, argv
        else:
            assert "=" in arr[0], f"{arr[0]} requires a value."
            argv.remove(arr[0])
            return arr[0].split("=")[1], argv


no_debug, argv = _argparse("--nodebug", argv)

USE_NINJA = os.getenv("USE_NINJA") == "0"
HERE = Path(os.path.dirname(__file__)).absolute()
SRC_PATH = HERE.parent.parent / "src"
CXX = os.environ["CXX"]

if sys.platform == "win32":
    vc_version = os.getenv("VCToolsVersion", "")
    if vc_version.startswith("14.16."):
        CXX_FLAGS = ["/sdl"]
    else:
        CXX_FLAGS = ["/sdl", "/permissive-"]
else:
    CXX_FLAGS = []

NVCC_FLAGS = [f"-ccbin={CXX}", "--extended-lambda"]

if not no_debug:
    CXX_FLAGS += ["-g", "-DDEBUG"]
    NVCC_FLAGS += ["-g", "-DDEBUG"]
else:
    CXX_FLAGS += ["-O3"]
    NVCC_FLAGS += ["-O3"]

ext_modules = [
    CUDAExtension(
        name="torch_cusparse._C",
        sources=["src/cusparse11.cu"],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS,},
        libraries=["cusparse"],
    ),
]

setup(
    name="torch_cusparse",
    packages=[],
    ext_modules=ext_modules,
    include_dirs=["src"],
    cmdclass={"build_ext": BuildExtension},
)
