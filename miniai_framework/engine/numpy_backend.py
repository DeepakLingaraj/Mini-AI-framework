"""
High-performance NumPy vectorized inference engine with KV-Cache optimization.
"""

import time
import numpy as np
from typing import Generator, Optional, List
from miniai_framework.config import RuntimeConfig
from miniai_framework.models.base import BaseModel
from miniai_framework.tokenizer.base import BaseTokenizer
from miniai_framework.engine.backend import BaseInferenceEngine, GenerationChunk
from miniai_framework.engine.sampler import LogitSampler


class NumPyInferenceEngine(BaseInferenceEngine):
    def generate_stream(
        self,
        prompt: str,
        runtime_config: Optional[RuntimeConfig] = None
    ) -> Generator[GenerationChunk, None, None]:
        if runtime_config is None:
            runtime_config = RuntimeConfig()

        if runtime_config.seed is not None:
            np.random.seed(runtime_config.seed)

        # 1. Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(input_ids) == 0:
            input_ids = [self.model.config.bos_token_id]

        start_time = time.perf_counter()
        kv_cache = {}
        generated_tokens: List[int] = []

        curr_input = np.array([input_ids], dtype=np.int32)
        start_pos = 0

        # Initial prompt pass
        step_start = time.perf_counter()
        logits, kv_cache = self.model.forward(curr_input, start_pos=start_pos, kv_cache=kv_cache)
        step_end = time.perf_counter()

        last_logits = logits[0, -1, :] # Logits of last token in prompt
        next_token_id = LogitSampler.sample(
            logits=last_logits,
            temperature=runtime_config.temperature,
            top_k=runtime_config.top_k,
            top_p=runtime_config.top_p,
            generated_tokens=generated_tokens,
            repetition_penalty=runtime_config.repetition_penalty
        )

        generated_tokens.append(next_token_id)
        start_pos += len(input_ids)

        token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
        ttft_ms = (step_end - start_time) * 1000.0

        yield GenerationChunk(
            token_id=next_token_id,
            token_text=token_text,
            is_finished=False,
            step=1,
            latency_ms=ttft_ms,
            total_time_ms=ttft_ms
        )

        # 2. Auto-regressive generation loop using KV-Cache
        for step in range(2, runtime_config.max_new_tokens + 1):
            if next_token_id == self.model.config.eos_token_id:
                break

            step_start = time.perf_counter()
            # Feed only single token input when using KV cache
            curr_input = np.array([[next_token_id]], dtype=np.int32)

            logits, kv_cache = self.model.forward(
                curr_input,
                start_pos=start_pos,
                kv_cache=kv_cache if runtime_config.use_kv_cache else None
            )

            last_logits = logits[0, -1, :]
            next_token_id = LogitSampler.sample(
                logits=last_logits,
                temperature=runtime_config.temperature,
                top_k=runtime_config.top_k,
                top_p=runtime_config.top_p,
                generated_tokens=generated_tokens,
                repetition_penalty=runtime_config.repetition_penalty
            )

            step_end = time.perf_counter()
            generated_tokens.append(next_token_id)
            start_pos += 1

            latency_ms = (step_end - step_start) * 1000.0
            total_time_ms = (step_end - start_time) * 1000.0
            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            is_eos = (next_token_id == self.model.config.eos_token_id) or (step == runtime_config.max_new_tokens)

            yield GenerationChunk(
                token_id=next_token_id,
                token_text=token_text,
                is_finished=is_eos,
                step=step,
                latency_ms=latency_ms,
                total_time_ms=total_time_ms
            )

            if is_eos:
                break
