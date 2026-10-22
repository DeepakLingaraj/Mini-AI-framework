# Mojo Scaled Dot-Product Attention Kernel
# Mini AI Framework - Vectorized Multi-Head Attention

from algorithm import parallelize, vectorize
from sys.info import simdbitwidth
from memory import UnsafePointer
from math import sqrt, exp

alias dtype = DType.float32
alias simd_width = simdbitwidth[dtype]()

fn scaled_dot_product_attention_mojo(
    Q_ptr: UnsafePointer[Scalar[dtype]],
    K_ptr: UnsafePointer[Scalar[dtype]],
    V_ptr: UnsafePointer[Scalar[dtype]],
    Out_ptr: UnsafePointer[Scalar[dtype]],
    batch_size: Int,
    n_heads: Int,
    seq_len: Int,
    head_dim: Int
):
    """
    SIMD-accelerated Attention kernel: Softmax((Q x K^T) / sqrt(head_dim)) x V.
    """
    let scale = 1.0 / sqrt(Float32(head_dim))
    
    @parameter
    fn compute_head(head_idx: Int):
        let b = head_idx / n_heads
        let h = head_idx % n_heads
        let head_offset = (b * n_heads + h) * seq_len * head_dim

        for i in range(seq_len):
            let q_offset = head_offset + i * head_dim
            
            # 1. Compute attention scores for row i
            var max_score: Float32 = -1e9
            for j in range(seq_len):
                let k_offset = head_offset + j * head_dim
                var dot: Float32 = 0.0
                
                # SIMD dot product Q[i] . K[j]
                for d in range(0, head_dim, simd_width):
                    let q_vec = Q_ptr.load[width=simd_width](q_offset + d)
                    let k_vec = K_ptr.load[width=simd_width](k_offset + d)
                    dot += (q_vec * k_vec).reduce_add()
                
                let score = dot * scale
                if score > max_score:
                    max_score = score

            # 2. Softmax normalization and weighted sum over V
            for d in range(0, head_dim, simd_width):
                var acc = SIMD[dtype, simd_width](0.0)
                var norm: Float32 = 0.0
                
                for j in range(seq_len):
                    let k_offset = head_offset + j * head_dim
                    let v_offset = head_offset + j * head_dim
                    
                    var dot: Float32 = 0.0
                    for k_d in range(0, head_dim, simd_width):
                        dot += (Q_ptr.load[width=simd_width](q_offset + k_d) * K_ptr.load[width=simd_width](k_offset + k_d)).reduce_add()
                    
                    let weight = exp(dot * scale - max_score)
                    norm += weight
                    let v_vec = V_ptr.load[width=simd_width](v_offset + d)
                    acc += weight * v_vec

                let out_offset = head_offset + i * head_dim + d
                Out_ptr.store[width=simd_width](out_offset, acc / norm)

    parallelize[compute_head](batch_size * n_heads, batch_size * n_heads)

fn main():
    print("Mojo Multi-Head Scaled Dot-Product Attention Kernel Loaded.")
