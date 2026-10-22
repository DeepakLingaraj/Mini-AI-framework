"""
Configuration definitions and data structures for Mini AI Framework.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any


class BackendType(str, Enum):
    NUMPY = "numpy"
    MAX_ENGINE = "max_engine"
    MOJO_SIMD = "mojo_simd"
    AUTO = "auto"


class QuantType(str, Enum):
    NONE = "none"
    INT8 = "int8"
    FP16 = "fp16"


@dataclass
class QuantConfig:
    enabled: bool = False
    mode: QuantType = QuantType.NONE
    scale: float = 1.0
    zero_point: int = 0


@dataclass
class ModelConfig:
    name: str = "MiniLlama-15M"
    architecture: str = "NanoTransformer"
    vocab_size: int = 4096
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 684
    max_seq_len: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int = 3
    quantization: QuantConfig = field(default_factory=QuantConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            name=data.get("name", "MiniLlama-15M"),
            architecture=data.get("architecture", "NanoTransformer"),
            vocab_size=data.get("vocab_size", 4096),
            d_model=data.get("d_model", 256),
            n_heads=data.get("n_heads", 8),
            n_layers=data.get("n_layers", 6),
            d_ff=data.get("d_ff", 684),
            max_seq_len=data.get("max_seq_len", 512),
            rope_theta=data.get("rope_theta", 10000.0),
            bos_token_id=data.get("bos_token_id", 1),
            eos_token_id=data.get("eos_token_id", 2),
            pad_token_id=data.get("pad_token_id", 0),
            unk_token_id=data.get("unk_token_id", 3),
        )


@dataclass
class RuntimeConfig:
    backend: BackendType = BackendType.AUTO
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    seed: Optional[int] = 42
    use_kv_cache: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.value,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "use_kv_cache": self.use_kv_cache,
        }
