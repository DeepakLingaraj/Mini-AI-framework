"""
Byte-Pair Encoding (BPE) Tokenizer implementation.
"""

import json
import re
from typing import List, Dict, Tuple, Optional, Set
from miniai_framework.tokenizer.base import BaseTokenizer, TokenPiece


class BPETokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self.special_tokens = special_tokens or {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}

        if vocab is not None:
            self.vocab = dict(vocab)
            self.merges = list(merges) if merges else []
        else:
            self._init_default_vocab()

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

    def _init_default_vocab(self):
        """Build clean default BPE vocabulary containing common ASCII and AI terms."""
        self.vocab = {}
        self.merges = []

        # Add special tokens
        for token, token_id in self.special_tokens.items():
            self.vocab[token] = token_id

        curr_id = len(self.vocab)

        # Single byte / character tokens
        for c in range(256):
            ch = chr(c)
            if ch not in self.vocab:
                self.vocab[ch] = curr_id
                curr_id += 1

        # Common BPE subwords & tokens
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
            "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
            "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
            "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "model", "token", "inference", "benchmark", "ai", "max", "mojo",
            "python", "framework", "layer", "attention", "transformer", "latency",
            "throughput", "tensor", "matrix", "vector", "speed", "fast", "gpu",
            "cpu", "memory", "int8", "fp16", "fp32", "llm", "neural", "network"
        ]

        for word in common_words:
            # Add character pairs merge
            chars = list(word)
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i+1])
                if pair not in self.merges:
                    self.merges.append(pair)
            if word not in self.vocab:
                self.vocab[word] = curr_id
                curr_id += 1

    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe_encode_word(self, word: str) -> List[str]:
        if len(word) == 0:
            return []
        if len(word) == 1:
            return [word]

        word_pieces = list(word)
        while len(word_pieces) > 1:
            pairs = self._get_pairs(word_pieces)
            min_pair = None
            min_rank = float("inf")

            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            if min_pair is None or min_pair not in self.bpe_ranks:
                break

            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word_pieces):
                if i < len(word_pieces) - 1 and word_pieces[i] == first and word_pieces[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word_pieces[i])
                    i += 1
            word_pieces = new_word

        return word_pieces

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if not text:
            return []

        tokens = []
        if add_special_tokens:
            tokens.append(self.special_tokens["<bos>"])

        # Split text into regex tokens (words, digits, punctuation, whitespace)
        words = re.findall(r'\s+|\w+|[^\w\s]', text)

        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                pieces = self._bpe_encode_word(word)
                for p in pieces:
                    if p in self.vocab:
                        tokens.append(self.vocab[p])
                    else:
                        # Character level fallback or <unk>
                        for ch in p:
                            tokens.append(self.vocab.get(ch, self.special_tokens["<unk>"]))

        if add_special_tokens:
            tokens.append(self.special_tokens["<eos>"])

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        text_pieces = []
        for tid in token_ids:
            if skip_special_tokens and tid in self.inverse_special:
                continue
            if tid in self.inverse_vocab:
                text_pieces.append(self.inverse_vocab[tid])
            elif tid in self.inverse_special:
                text_pieces.append(f"[{self.inverse_special[tid]}]")
            else:
                text_pieces.append("<unk>")

        return "".join(text_pieces)

    def tokenize_with_metadata(self, text: str) -> List[TokenPiece]:
        token_ids = self.encode(text, add_special_tokens=False)
        pieces = []
        curr_pos = 0

        for tid in token_ids:
            token_str = self.inverse_vocab.get(tid, "<unk>")
            is_special = tid in self.inverse_special

            # Find matching char span in original text
            start = text.find(token_str, curr_pos) if not is_special else curr_pos
            if start == -1:
                start = curr_pos
            end = start + len(token_str)
            curr_pos = end

            pieces.append(TokenPiece(
                id=tid,
                text=token_str,
                start_char=start,
                end_char=end,
                is_special=is_special
            ))

        return pieces

    def vocab_size(self) -> int:
        return len(self.vocab)

    def save_vocab(self, filepath: str):
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_vocab(cls, filepath: str) -> "BPETokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = [tuple(pair) for pair in data.get("merges", [])]
        return cls(
            vocab=data.get("vocab"),
            merges=merges,
            special_tokens=data.get("special_tokens")
        )
