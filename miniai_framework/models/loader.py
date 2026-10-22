"""
Model loading and serialization utilities.
"""

import json
import os
from typing import Dict, Any, Optional
import numpy as np
from miniai_framework.config import ModelConfig
from miniai_framework.models.base import BaseModel
from miniai_framework.models.transformer import NanoTransformer


class ModelLoader:
    @staticmethod
    def create_synthetic_model(config: Optional[ModelConfig] = None) -> NanoTransformer:
        """Create a NanoTransformer instance initialized with synthetic weights."""
        if config is None:
            config = ModelConfig()
        return NanoTransformer(config=config)

    @staticmethod
    def save_model(model: BaseModel, output_path: str):
        """Save model configuration and weights to directory."""
        os.makedirs(output_path, exist_ok=True)
        config_file = os.path.join(output_path, "config.json")
        weights_file = os.path.join(output_path, "weights.npz")

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        np.savez_compressed(weights_file, **model.weights)

    @staticmethod
    def load_model(model_dir: str) -> NanoTransformer:
        """Load model configuration and weights from directory."""
        config_file = os.path.join(model_dir, "config.json")
        weights_file = os.path.join(model_dir, "weights.npz")

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found at {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config = ModelConfig.from_dict(config_dict)

        if os.path.exists(weights_file):
            weights_data = np.load(weights_file)
            weights = {key: weights_data[key] for key in weights_data.files}
        else:
            weights = None

        return NanoTransformer(config=config, weights=weights)
