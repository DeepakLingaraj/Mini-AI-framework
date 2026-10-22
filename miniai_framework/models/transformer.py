"""
NanoTransformer architecture implementation using vectorized NumPy operations.
Features RoPE, RMSNorm, SwiGLU, and KV Cache indexing for fast auto-regressive generation.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from miniai_framework.config import ModelConfig
from miniai_framework.models.base import BaseModel


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Root Mean Square Normalization."""
    variance = np.mean(np.square(x), axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)) * weight


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation function: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -100, 100))))


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute Rotary Positional Embedding (RoPE) cos and sin frequencies."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(end, dtype=np.float32)
    freqs_outer = np.outer(t, freqs)
    cos = np.cos(freqs_outer)
    sin = np.sin(freqs_outer)
    return cos, sin


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, start_pos: int) -> np.ndarray:
    """
    Apply Rotary Positional Embedding to query/key tensors.
    x shape: (batch, seq_len, n_heads, head_dim)
    """
    seq_len = x.shape[1]
    head_dim = x.shape[3]
    half_dim = head_dim // 2

    # Slice cos/sin for current positions
    cos_pos = cos[start_pos : start_pos + seq_len, :half_dim] # (seq_len, half_dim)
    sin_pos = sin[start_pos : start_pos + seq_len, :half_dim]

    # Reshape for broadcasting over heads: (1, seq_len, 1, half_dim)
    cos_pos = np.expand_dims(np.expand_dims(cos_pos, axis=0), axis=2)
    sin_pos = np.expand_dims(np.expand_dims(sin_pos, axis=0), axis=2)

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    out1 = x1 * cos_pos - x2 * sin_pos
    out2 = x1 * sin_pos + x2 * cos_pos

    return np.concatenate([out1, out2], axis=-1)


class NanoTransformer(BaseModel):
    def __init__(self, config: ModelConfig, weights: Optional[Dict[str, np.ndarray]] = None):
        super().__init__(config)
        self.head_dim = config.d_model // config.n_heads

        # Precompute RoPE freqs
        self.cos_freqs, self.sin_freqs = precompute_freqs_cis(
            dim=self.head_dim,
            end=config.max_seq_len * 2,
            theta=config.rope_theta
        )

        if weights is not None:
            self.weights = weights
        else:
            self.init_weights()

    def init_weights(self):
        """Initialize random synthetic weights for testing when no external weights loaded."""
        rng = np.random.RandomState(42)
        scale = 0.02
        c = self.config

        # Embedding & LM Head
        self.weights["embed_tokens"] = rng.normal(0, scale, (c.vocab_size, c.d_model)).astype(np.float32)
        self.weights["norm.weight"] = np.ones(c.d_model, dtype=np.float32)
        self.weights["lm_head"] = rng.normal(0, scale, (c.d_model, c.vocab_size)).astype(np.float32)

        # Transformer Layers
        for l in range(c.n_layers):
            prefix = f"layers.{l}."
            self.weights[prefix + "input_norm.weight"] = np.ones(c.d_model, dtype=np.float32)
            self.weights[prefix + "post_norm.weight"] = np.ones(c.d_model, dtype=np.float32)

            # Attention projections (Q, K, V, O)
            self.weights[prefix + "attn.wq"] = rng.normal(0, scale, (c.d_model, c.d_model)).astype(np.float32)
            self.weights[prefix + "attn.wk"] = rng.normal(0, scale, (c.d_model, c.d_model)).astype(np.float32)
            self.weights[prefix + "attn.wv"] = rng.normal(0, scale, (c.d_model, c.d_model)).astype(np.float32)
            self.weights[prefix + "attn.wo"] = rng.normal(0, scale, (c.d_model, c.d_model)).astype(np.float32)

            # MLP SwiGLU projections (w1=gate, w2=down, w3=up)
            self.weights[prefix + "mlp.w1"] = rng.normal(0, scale, (c.d_model, c.d_ff)).astype(np.float32)
            self.weights[prefix + "mlp.w2"] = rng.normal(0, scale, (c.d_ff, c.d_model)).astype(np.float32)
            self.weights[prefix + "mlp.w3"] = rng.normal(0, scale, (c.d_model, c.d_ff)).astype(np.float32)

    def forward(
        self,
        input_ids: np.ndarray,
        start_pos: int = 0,
        kv_cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through NanoTransformer.
        input_ids: (batch_size, seq_len)
        """
        b, seq_len = input_ids.shape
        c = self.config

        # 1. Embedding lookup
        h = self.weights["embed_tokens"][input_ids] # (b, seq_len, d_model)

        if kv_cache is None:
            kv_cache = {}

        # 2. Iterate through Transformer Blocks
        for l in range(c.n_layers):
            prefix = f"layers.{l}."

            # Layer Norm 1
            norm_h = rms_norm(h, self.weights[prefix + "input_norm.weight"], c.layer_norm_eps)

            # Q, K, V linear projections
            q = np.matmul(norm_h, self.weights[prefix + "attn.wq"])
            k = np.matmul(norm_h, self.weights[prefix + "attn.wk"])
            v = np.matmul(norm_h, self.weights[prefix + "attn.wv"])

            # Reshape into multi-head: (b, seq_len, n_heads, head_dim)
            q = q.reshape(b, seq_len, c.n_heads, self.head_dim)
            k = k.reshape(b, seq_len, c.n_heads, self.head_dim)
            v = v.reshape(b, seq_len, c.n_heads, self.head_dim)

            # Apply RoPE
            q = apply_rope(q, self.cos_freqs, self.sin_freqs, start_pos)
            k = apply_rope(k, self.cos_freqs, self.sin_freqs, start_pos)

            # Update KV-Cache
            cache_k_key = f"layer_{l}_k"
            cache_v_key = f"layer_{l}_v"

            if cache_k_key in kv_cache:
                k = np.concatenate([kv_cache[cache_k_key], k], axis=1)
                v = np.concatenate([kv_cache[cache_v_key], v], axis=1)

            kv_cache[cache_k_key] = k
            kv_cache[cache_v_key] = v

            # Scaled Dot-Product Attention
            # Transpose for batched matmul: (b, n_heads, seq_len, head_dim)
            q_t = np.transpose(q, (0, 2, 1, 3))
            k_t = np.transpose(k, (0, 2, 1, 3))
            v_t = np.transpose(v, (0, 2, 1, 3))

            # Scores: (b, n_heads, q_len, kv_len)
            scores = np.matmul(q_t, np.transpose(k_t, (0, 1, 3, 2))) / np.sqrt(self.head_dim)

            # Apply causal mask if initial prompt pass
            kv_seq_len = k_t.shape[2]
            if seq_len > 1:
                mask = np.triu(np.full((seq_len, kv_seq_len), -1e9, dtype=np.float32), k=1)
                scores = scores + mask

            # Softmax
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            # Output attention context
            output_attn = np.matmul(attn_weights, v_t) # (b, n_heads, q_len, head_dim)
            output_attn = np.transpose(output_attn, (0, 2, 1, 3)).reshape(b, seq_len, c.d_model)

            # Output projection + residual connection
            attn_out = np.matmul(output_attn, self.weights[prefix + "attn.wo"])
            h = h + attn_out

            # Layer Norm 2 + SwiGLU MLP
            mlp_norm_h = rms_norm(h, self.weights[prefix + "post_norm.weight"], c.layer_norm_eps)
            w1_out = np.matmul(mlp_norm_h, self.weights[prefix + "mlp.w1"]) # gate
            w3_out = np.matmul(mlp_norm_h, self.weights[prefix + "mlp.w3"]) # up
            mlp_act = silu(w1_out) * w3_out
            mlp_out = np.matmul(mlp_act, self.weights[prefix + "mlp.w2"]) # down

            h = h + mlp_out

        # Final RMSNorm & LM Head projection
        h = rms_norm(h, self.weights["norm.weight"], c.layer_norm_eps)
        logits = np.matmul(h, self.weights["lm_head"])

        return logits, kv_cache
