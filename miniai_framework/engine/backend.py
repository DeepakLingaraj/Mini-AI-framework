"""
Base Inference Engine and streaming result datatypes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, List, Dict, Any, Optional
import time
from miniai_framework.config import RuntimeConfig
from miniai_framework.models.base import BaseModel
from miniai_framework.tokenizer.base import BaseTokenizer


@dataclass
class GenerationChunk:
    token_id: int
    token_text: str
    is_finished: bool
    step: int
    latency_ms: float
    total_time_ms: float


@dataclass
class GenerationResult:
    text: str
    tokens: List[int]
    total_time_ms: float
    ttft_ms: float # Time to first token
    tokens_per_second: float
    avg_latency_per_token_ms: float


class BaseInferenceEngine(ABC):
    def __init__(self, model: BaseModel, tokenizer: BaseTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        runtime_config: Optional[RuntimeConfig] = None
    ) -> Generator[GenerationChunk, None, None]:
        """Stream generated tokens one by one."""
        pass

    def generate(
        self,
        prompt: str,
        runtime_config: Optional[RuntimeConfig] = None
    ) -> GenerationResult:
        """Run complete generation and return full metrics."""
        chunks = list(self.generate_stream(prompt, runtime_config))
        if not chunks:
            return GenerationResult("", [], 0.0, 0.0, 0.0, 0.0)

        full_text = "".join(chunk.token_text for chunk in chunks)
        token_ids = [chunk.token_id for chunk in chunks]
        total_time = chunks[-1].total_time_ms
        ttft = chunks[0].total_time_ms if len(chunks) > 0 else 0.0
        num_tokens = len(chunks)
        tps = (num_tokens / (total_time / 1000.0)) if total_time > 0 else 0.0
        avg_latency = total_time / max(num_tokens, 1)

        return GenerationResult(
            text=full_text,
            tokens=token_ids,
            total_time_ms=total_time,
            ttft_ms=ttft,
            tokens_per_second=tps,
            avg_latency_per_token_ms=avg_latency
        )
