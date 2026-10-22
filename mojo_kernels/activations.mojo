# Mojo SIMD Activations and Normalization Kernel
# Mini AI Framework - Vectorized RMSNorm & SwiGLU Functions

from algorithm import vectorize
from sys.info import simdbitwidth
from memory import UnsafePointer
from math import sqrt, exp

alias dtype = DType.float32
alias simd_width = simdbitwidth[dtype]()

fn rms_norm_mojo(
    x_ptr: UnsafePointer[Scalar[dtype]],
    weight_ptr: UnsafePointer[Scalar[dtype]],
    out_ptr: UnsafePointer[Scalar[dtype]],
    size: Int,
    eps: Float32 = 1e-5
):
    """
    Vectorized Root Mean Square Normalization in Mojo.
    """
    var sum_sq: Float32 = 0.0
    for i in range(0, size, simd_width):
        let vec = x_ptr.load[width=simd_width](i)
        sum_sq += (vec * vec).reduce_add()
    
    let inv_rms = 1.0 / sqrt((sum_sq / Float32(size)) + eps)
    
    for i in range(0, size, simd_width):
        let vec = x_ptr.load[width=simd_width](i)
        let w_vec = weight_ptr.load[width=simd_width](i)
        out_ptr.store[width=simd_width](i, vec * inv_rms * w_vec)

fn swiglu_mojo(
    gate_ptr: UnsafePointer[Scalar[dtype]],
    up_ptr: UnsafePointer[Scalar[dtype]],
    out_ptr: UnsafePointer[Scalar[dtype]],
    size: Int
):
    """
    Vectorized SwiGLU Activation: SiLU(gate) * up.
    """
    for i in range(0, size, simd_width):
        let g_vec = gate_ptr.load[width=simd_width](i)
        let u_vec = up_ptr.load[width=simd_width](i)
        
        # SiLU(x) = x / (1 + exp(-x))
        let silu_vec = g_vec / (SIMD[dtype, simd_width](1.0) + (-g_vec).exp())
        out_ptr.store[width=simd_width](i, silu_vec * u_vec)

fn main():
    print("Mojo Vectorized RMSNorm & SwiGLU Activations Kernel Loaded.")
