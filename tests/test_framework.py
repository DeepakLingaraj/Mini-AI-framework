"""
Comprehensive Unit Test Suite for Mini AI Framework.
"""

import sys
import os
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from miniai_framework.config import ModelConfig, RuntimeConfig, BackendType
from miniai_framework.tokenizer import BPETokenizer
from miniai_framework.models import ModelLoader, NanoTransformer
from miniai_framework.engine import NumPyInferenceEngine, LogitSampler
from miniai_framework.benchmark import BenchmarkProfiler, BackendComparator


class TestMiniAIFramework(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BPETokenizer()
        self.config = ModelConfig(vocab_size=self.tokenizer.vocab_size(), d_model=64, n_heads=4, n_layers=2)
        self.model = NanoTransformer(config=self.config)

    def test_tokenizer_encode_decode(self):
        text = "Mini AI framework test"
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        self.assertGreater(len(tokens), 0)
        self.assertEqual(tokens[0], self.tokenizer.special_tokens["<bos>"])
        self.assertEqual(tokens[-1], self.tokenizer.special_tokens["<eos>"])

        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self.assertIn("AI", decoded)

    def test_tokenizer_metadata(self):
        pieces = self.tokenizer.tokenize_with_metadata("Modular MAX Mojo")
        self.assertGreater(len(pieces), 0)
        self.assertTrue(hasattr(pieces[0], "start_char"))
        self.assertTrue(hasattr(pieces[0], "end_char"))

    def test_model_forward_pass(self):
        input_ids = np.array([[1, 10, 25]], dtype=np.int32)
        logits, kv_cache = self.model.forward(input_ids, start_pos=0)
        self.assertEqual(logits.shape, (1, 3, self.config.vocab_size))
        self.assertIsNotNone(kv_cache)

    def test_logit_sampler(self):
        logits = np.random.randn(self.config.vocab_size)
        sample_id = LogitSampler.sample(logits, temperature=0.7, top_k=10, top_p=0.9)
        self.assertTrue(0 <= sample_id < self.config.vocab_size)

    def test_numpy_inference_engine_stream(self):
        engine = NumPyInferenceEngine(self.model, self.tokenizer)
        cfg = RuntimeConfig(max_new_tokens=10, temperature=0.7)
        chunks = list(engine.generate_stream("AI framework", cfg))
        self.assertGreater(len(chunks), 0)
        self.assertTrue(chunks[-1].is_finished or len(chunks) == 10)

    def test_benchmark_profiler(self):
        metrics = BenchmarkProfiler.run_benchmark(
            model=self.model,
            tokenizer=self.tokenizer,
            backend_type=BackendType.NUMPY,
            prompt="Test prompt",
            max_new_tokens=15,
            warmup_runs=0
        )
        self.assertGreater(metrics.tokens_per_second, 0.0)
        self.assertGreater(metrics.memory_footprint_mb, 0.0)


if __name__ == "__main__":
    unittest.main()
