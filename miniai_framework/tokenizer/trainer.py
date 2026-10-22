"""
Tokenizer Trainer for training BPE tokenizers from raw text corpora.
"""

import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
from miniai_framework.tokenizer.bpe import BPETokenizer


class BPETrainer:
    def __init__(self, target_vocab_size: int = 1000):
        self.target_vocab_size = target_vocab_size

    def train(self, texts: List[str]) -> BPETokenizer:
        special_tokens = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        vocab = dict(special_tokens)
        curr_id = len(vocab)

        # Tokenize corpus into word frequencies
        word_counts = Counter()
        for text in texts:
            words = re.findall(r'\w+|\S', text)
            for w in words:
                spaced_word = " ".join(list(w))
                word_counts[spaced_word] += 1

        # Add all unique characters to vocab
        all_chars = set()
        for word in word_counts.keys():
            all_chars.update(word.split())

        for ch in sorted(all_chars):
            if ch not in vocab:
                vocab[ch] = curr_id
                curr_id += 1

        merges: List[Tuple[str, str]] = []

        # Perform BPE merges
        num_merges = self.target_vocab_size - len(vocab)
        for _ in range(max(0, num_merges)):
            pairs = defaultdict(int)
            for word, freq in word_counts.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            merges.append(best_pair)
            merged_token = "".join(best_pair)

            if merged_token not in vocab:
                vocab[merged_token] = curr_id
                curr_id += 1

            # Update word counts with merged token
            new_word_counts = {}
            bigram = f"{best_pair[0]} {best_pair[1]}"
            replacement = merged_token

            for word, freq in word_counts.items():
                new_word = word.replace(bigram, replacement)
                new_word_counts[new_word] = freq
            word_counts = new_word_counts

        return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
