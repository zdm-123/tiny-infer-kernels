"""tiny-infer-kernels: Triton operators for LLM inference."""

from tiny_infer_kernels.kernels.flash_attn import flash_attn
from tiny_infer_kernels.kernels.flash_decode import flash_decode
from tiny_infer_kernels.kernels.rmsnorm import rmsnorm
from tiny_infer_kernels.kernels.rope import rope
from tiny_infer_kernels.kernels.softmax import softmax
from tiny_infer_kernels.kernels.swiglu import swiglu
from tiny_infer_kernels.kernels.w8a16_gemm import w8a16_gemm

__all__ = ["flash_attn", "flash_decode", "rmsnorm", "rope", "softmax", "swiglu", "w8a16_gemm"]
__version__ = "0.1.0"
