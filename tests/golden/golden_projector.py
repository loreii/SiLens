#!/usr/bin/env python3
"""
SiLens Golden Model: Multimodal Projector
==========================================

Golden model for the vision-to-language projector.

This module provides:
- Linear projection golden model
- Two-layer MLP projector
- Test vector generation

License: Apache 2.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .golden_attention import WeightType, PrecisionMode, FixedPointOps, TestVector


@dataclass
class ProjectorConfig:
    """Configuration for multimodal projector."""
    vision_dim: int = 768       # Input dim from vision encoder
    language_dim: int = 576     # Output dim for language model
    hidden_dim: int = 2048      # Hidden dimension (for MLP projector)
    projector_type: str = "mlp" # 'linear' or 'mlp'
    act_width: int = 8
    frac_bits: int = 4


class GoldenLinearProjector:
    """Golden model for simple linear projection."""
    
    def __init__(
        self,
        config: ProjectorConfig,
        weight_type: WeightType = WeightType.TERNARY
    ):
        self.config = config
        self.weight_type = weight_type
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        np.random.seed(200)
        
        if self.weight_type == WeightType.TERNARY:
            self.weight = np.random.choice(
                [-1, 0, 1],
                size=(self.config.language_dim, self.config.vision_dim),
                p=[0.35, 0.3, 0.35]
            ).astype(np.int8)
            self.scale = 0.02
        else:
            self.weight = np.random.randn(
                self.config.language_dim, self.config.vision_dim
            ).astype(np.float32) * 0.02
            self.scale = 1.0
        
        self.bias = np.zeros(self.config.language_dim, dtype=np.float32)
    
    def _ternary_matmul(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Ternary matrix multiplication."""
        pos_mask = (w == 1)
        neg_mask = (w == -1)
        result = np.zeros((x.shape[0], w.shape[0]), dtype=np.float64)
        
        for i in range(w.shape[0]):
            pos_sum = np.sum(x[:, pos_mask[i]], axis=-1)
            neg_sum = np.sum(x[:, neg_mask[i]], axis=-1)
            result[:, i] = pos_sum - neg_sum
        
        return (result * self.scale).astype(np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project vision features to language space."""
        if self.weight_type == WeightType.TERNARY:
            return self._ternary_matmul(x, self.weight) + self.bias
        return (x @ self.weight.T + self.bias).astype(np.float32)


class GoldenMLPProjector:
    """
    Golden model for MLP projector.
    
    Architecture: vision_dim -> hidden_dim -> language_dim
    """
    
    def __init__(
        self,
        config: ProjectorConfig,
        weight_type: WeightType = WeightType.TERNARY
    ):
        self.config = config
        self.weight_type = weight_type
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights."""
        np.random.seed(201)
        
        if self.weight_type == WeightType.TERNARY:
            self.W1 = np.random.choice(
                [-1, 0, 1],
                size=(self.config.hidden_dim, self.config.vision_dim),
                p=[0.35, 0.3, 0.35]
            ).astype(np.int8)
            self.W2 = np.random.choice(
                [-1, 0, 1],
                size=(self.config.language_dim, self.config.hidden_dim),
                p=[0.35, 0.3, 0.35]
            ).astype(np.int8)
            self.scale1 = 0.02
            self.scale2 = 0.02
        else:
            self.W1 = np.random.randn(
                self.config.hidden_dim, self.config.vision_dim
            ).astype(np.float32) * 0.02
            self.W2 = np.random.randn(
                self.config.language_dim, self.config.hidden_dim
            ).astype(np.float32) * 0.02
            self.scale1 = 1.0
            self.scale2 = 1.0

    def _ternary_matmul(self, x: np.ndarray, w: np.ndarray, scale: float) -> np.ndarray:
        """Ternary matrix multiplication."""
        pos_mask = (w == 1)
        neg_mask = (w == -1)
        result = np.zeros((x.shape[0], w.shape[0]), dtype=np.float64)
        
        for i in range(w.shape[0]):
            pos_sum = np.sum(x[:, pos_mask[i]], axis=-1)
            neg_sum = np.sum(x[:, neg_mask[i]], axis=-1)
            result[:, i] = pos_sum - neg_sum
        
        return (result * scale).astype(np.float32)
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return (0.5 * x * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        ))).astype(np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through MLP projector."""
        if self.weight_type == WeightType.TERNARY:
            hidden = self._ternary_matmul(x, self.W1, self.scale1)
        else:
            hidden = x @ self.W1.T
        
        hidden = self._gelu(hidden)
        
        if self.weight_type == WeightType.TERNARY:
            output = self._ternary_matmul(hidden, self.W2, self.scale2)
        else:
            output = hidden @ self.W2.T
        
        return output


class GoldenMultimodalProjector:
    """
    Complete multimodal projector golden model.
    
    Maps vision encoder output to language model input space.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        language_dim: int = 576,
        hidden_dim: int = 2048,
        projector_type: str = "mlp",
        weight_type: Union[str, WeightType] = 'ternary'
    ):
        if isinstance(weight_type, str):
            weight_type = WeightType(weight_type)
        
        self.config = ProjectorConfig(
            vision_dim=vision_dim,
            language_dim=language_dim,
            hidden_dim=hidden_dim,
            projector_type=projector_type
        )
        self.weight_type = weight_type
        
        if projector_type == "linear":
            self.projector = GoldenLinearProjector(self.config, weight_type)
        else:
            self.projector = GoldenMLPProjector(self.config, weight_type)
    
    def forward(self, vision_features: np.ndarray) -> np.ndarray:
        """
        Project vision features to language space.
        
        Args:
            vision_features: [seq_len, vision_dim]
            
        Returns:
            projected: [seq_len, language_dim]
        """
        return self.projector.forward(vision_features)
    
    def generate_test_vectors(
        self,
        num_vectors: int = 10,
        seq_lengths: List[int] = None,
        seed: int = 42
    ) -> List[TestVector]:
        """Generate test vectors for RTL verification."""
        if seq_lengths is None:
            seq_lengths = [1, 16, 64, 256]
        
        np.random.seed(seed)
        vectors = []

        # Edge cases
        # All zeros
        x_zeros = np.zeros((16, self.config.vision_dim), dtype=np.float32)
        out_zeros = self.forward(x_zeros)
        vectors.append(TestVector(
            name="projector_zeros",
            inputs={'x': x_zeros},
            expected_outputs={'output': out_zeros},
            description="All zero input"
        ))
        
        # All ones
        x_ones = np.ones((16, self.config.vision_dim), dtype=np.float32)
        out_ones = self.forward(x_ones)
        vectors.append(TestVector(
            name="projector_ones",
            inputs={'x': x_ones},
            expected_outputs={'output': out_ones},
            description="All ones input"
        ))
        
        # Random vectors
        for seq_len in seq_lengths:
            for i in range(num_vectors // len(seq_lengths)):
                x = np.random.randn(seq_len, self.config.vision_dim).astype(np.float32) * 0.1
                output = self.forward(x)
                
                vectors.append(TestVector(
                    name=f"projector_seq{seq_len}_{i}",
                    inputs={'x': x},
                    expected_outputs={'output': output},
                    description=f"Random input, seq_len={seq_len}"
                ))
        
        return vectors


def test_golden_projector():
    """Test the golden projector implementation."""
    print("Testing Golden Multimodal Projector")
    print("=" * 50)
    
    # Test linear projector
    print("\n--- Linear Projector ---")
    config = ProjectorConfig(vision_dim=64, language_dim=32, projector_type="linear")
    linear = GoldenLinearProjector(config, WeightType.TERNARY)
    x = np.random.randn(16, 64).astype(np.float32) * 0.1
    out = linear.forward(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    assert out.shape == (16, 32)

    # Test MLP projector
    print("\n--- MLP Projector ---")
    config = ProjectorConfig(vision_dim=64, language_dim=32, hidden_dim=128, projector_type="mlp")
    mlp = GoldenMLPProjector(config, WeightType.TERNARY)
    out = mlp.forward(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    assert out.shape == (16, 32)
    
    # Test complete projector
    print("\n--- Complete Projector ---")
    projector = GoldenMultimodalProjector(
        vision_dim=64,
        language_dim=32,
        hidden_dim=128,
        projector_type="mlp",
        weight_type='ternary'
    )
    out = projector.forward(x)
    print(f"Output: {out.shape}")
    
    # Test vector generation
    print("\n--- Test Vector Generation ---")
    vectors = projector.generate_test_vectors(num_vectors=8)
    print(f"Generated {len(vectors)} test vectors")
    
    print("\n✓ All projector tests passed!")


if __name__ == "__main__":
    test_golden_projector()
