"""
Mini AI Framework (ApexMini / MAX-Mojo Python Engine)
A lightweight modular framework for model loading, tokenization, multi-backend inference, and benchmarking.
"""

__version__ = "0.1.0"
__author__ = "ApexMini AI Team"

from miniai_framework.config import ModelConfig, RuntimeConfig, QuantConfig, BackendType

__all__ = [
    "ModelConfig",
    "RuntimeConfig",
    "QuantConfig",
    "BackendType",
]
