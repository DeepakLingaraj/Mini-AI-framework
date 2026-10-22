"""
Abstract base class for models and layers in Mini AI Framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from miniai_framework.config import ModelConfig


class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.weights: Dict[str, np.ndarray] = {}

    @abstractmethod
    def forward(
        self,
        input_ids: np.ndarray,
        start_pos: int = 0,
        kv_cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Execute forward pass.
        Args:
            input_ids: Array of shape (batch_size, seq_len)
            start_pos: Position index for auto-regressive generation & KV cache indexing.
            kv_cache: Key-Value cache mapping for fast sequence extension.
        Returns:
            Tuple of (logits of shape [batch_size, seq_len, vocab_size], updated_kv_cache)
        """
        pass

    def get_num_parameters(self) -> int:
        """Calculate total number of parameters in the model."""
        return sum(w.size for w in self.weights.values())

    def get_memory_footprint_mb(self) -> float:
        """Calculate weight memory size in Megabytes."""
        bytes_total = sum(w.nbytes for w in self.weights.values())
        return bytes_total / (1024 * 1024)
