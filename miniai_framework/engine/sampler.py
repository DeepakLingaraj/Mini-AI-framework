"""
Logit sampling techniques: Temperature, Top-K, Top-P (Nucleus), Repetition Penalty.
"""

import numpy as np
from typing import List, Optional


class LogitSampler:
    @staticmethod
    def sample(
        logits: np.ndarray,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        generated_tokens: Optional[List[int]] = None,
        repetition_penalty: float = 1.1
    ) -> int:
        """
        Sample token index from 1D logits vector of shape (vocab_size,).
        """
        logits = logits.astype(np.float64).copy()

        # Repetition penalty
        if generated_tokens and repetition_penalty != 1.0:
            for token_id in set(generated_tokens):
                if logits[token_id] < 0:
                    logits[token_id] *= repetition_penalty
                else:
                    logits[token_id] /= repetition_penalty

        # Greedy decoding if temperature is 0
        if temperature <= 0.0:
            return int(np.argmax(logits))

        # Apply temperature
        logits /= max(temperature, 1e-5)

        # Apply Top-K filtering
        if top_k > 0 and top_k < len(logits):
            indices_to_remove = logits < np.sort(logits)[-top_k]
            logits[indices_to_remove] = -1e9

        # Apply Top-P (Nucleus) filtering
        if 0.0 < top_p < 1.0:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            
            # Softmax on sorted logits
            exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
            probs = exp_logits / np.sum(exp_logits)
            cumulative_probs = np.cumsum(probs)

            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -1e9

        # Softmax to get probabilities
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        # Random categorical sampling
        token_id = np.random.choice(len(probs), p=probs)
        return int(token_id)
