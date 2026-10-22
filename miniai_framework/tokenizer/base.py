"""
Abstract Base Tokenizer and Token metadata structure for Mini AI Framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


@dataclass
class TokenPiece:
    id: int
    text: str
    start_char: int
    end_char: int
    is_special: bool = False


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode string text into token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back into string text."""
        pass

    @abstractmethod
    def tokenize_with_metadata(self, text: str) -> List[TokenPiece]:
        """Tokenize text returning detailed token piece metadata for visualization."""
        pass

    @abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        pass
