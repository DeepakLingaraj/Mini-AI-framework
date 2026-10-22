"""
Multi-Backend Comparator for comparing execution across Python NumPy, Mojo SIMD, and MAX Engine.
"""

from typing import List, Dict, Any
from miniai_framework.config import BackendType
from miniai_framework.models.base import BaseModel
from miniai_framework.tokenizer.base import BaseTokenizer
from miniai_framework.benchmark.profiler import BenchmarkProfiler, BenchmarkMetrics


class BackendComparator:
    @staticmethod
    def compare_all(
        model: BaseModel,
        tokenizer: BaseTokenizer,
        prompt: str = "Mini AI Framework comparative benchmarking engine",
        max_new_tokens: int = 40
    ) -> List[BenchmarkMetrics]:
        """Run benchmark comparison across all available framework backends."""
        backends = [BackendType.NUMPY, BackendType.MAX_ENGINE, BackendType.MOJO_SIMD]
        results = []

        for backend in backends:
            metrics = BenchmarkProfiler.run_benchmark(
                model=model,
                tokenizer=tokenizer,
                backend_type=backend,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                warmup_runs=1
            )
            results.append(metrics)

        return results
