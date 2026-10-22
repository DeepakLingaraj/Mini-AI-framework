"""
Microsecond-accurate performance profiler for inference latency, throughput, TTFT, and FLOPs.
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from miniai_framework.config import RuntimeConfig, BackendType
from miniai_framework.models.base import BaseModel
from miniai_framework.tokenizer.base import BaseTokenizer
from miniai_framework.engine import NumPyInferenceEngine, MAXInferenceEngine, MojoInferenceEngine


@dataclass
class BenchmarkMetrics:
    backend_name: str
    model_name: str
    num_tokens_generated: int
    total_time_ms: float
    ttft_ms: float
    tokens_per_second: float
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    memory_footprint_mb: float
    estimated_gflops: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "model": self.model_name,
            "tokens_generated": self.num_tokens_generated,
            "total_time_ms": round(self.total_time_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2),
            "tokens_per_sec": round(self.tokens_per_second, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "memory_mb": round(self.memory_footprint_mb, 2),
            "gflops": round(self.estimated_gflops, 2),
        }


class BenchmarkProfiler:
    @staticmethod
    def run_benchmark(
        model: BaseModel,
        tokenizer: BaseTokenizer,
        backend_type: BackendType = BackendType.NUMPY,
        prompt: str = "Mini AI Framework high speed inference benchmark test",
        max_new_tokens: int = 50,
        warmup_runs: int = 1
    ) -> BenchmarkMetrics:
        """Run single backend benchmark."""
        # Instantiate backend
        if backend_type == BackendType.MAX_ENGINE:
            engine = MAXInferenceEngine(model, tokenizer)
        elif backend_type == BackendType.MOJO_SIMD:
            engine = MojoInferenceEngine(model, tokenizer)
        else:
            engine = NumPyInferenceEngine(model, tokenizer)

        runtime_cfg = RuntimeConfig(
            backend=backend_type,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            use_kv_cache=True
        )

        # Warmup runs
        for _ in range(warmup_runs):
            _ = list(engine.generate_stream(prompt, runtime_cfg))

        # Benchmark run
        start_t = time.perf_counter()
        chunks = list(engine.generate_stream(prompt, runtime_cfg))
        end_t = time.perf_counter()

        if not chunks:
            return BenchmarkMetrics(
                backend_name=backend_type.value,
                model_name=model.config.name,
                num_tokens_generated=0,
                total_time_ms=0.0,
                ttft_ms=0.0,
                tokens_per_second=0.0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p99_latency_ms=0.0,
                memory_footprint_mb=model.get_memory_footprint_mb(),
                estimated_gflops=0.0
            )

        total_time_ms = (end_t - start_t) * 1000.0
        ttft_ms = chunks[0].latency_ms
        latencies = [c.latency_ms for c in chunks]

        num_tokens = len(chunks)
        tps = (num_tokens / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0
        avg_lat = total_time_ms / num_tokens
        p50 = float(np.percentile(latencies, 50))
        p99 = float(np.percentile(latencies, 99))

        # Rough FLOPs estimation for 1 Transformer layer: 2 * num_params * num_tokens
        flops_per_token = 2 * model.get_num_parameters()
        total_gflops = (flops_per_token * num_tokens) / (total_time_ms / 1000.0) / 1e9

        return BenchmarkMetrics(
            backend_name=backend_type.value,
            model_name=model.config.name,
            num_tokens_generated=num_tokens,
            total_time_ms=total_time_ms,
            ttft_ms=ttft_ms,
            tokens_per_second=tps,
            avg_latency_ms=avg_lat,
            p50_latency_ms=p50,
            p99_latency_ms=p99,
            memory_footprint_mb=model.get_memory_footprint_mb(),
            estimated_gflops=total_gflops
        )
