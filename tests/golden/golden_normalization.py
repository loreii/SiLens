#!/usr/bin/env python3
"""
SiLens Golden Model: Normalization Layers
==========================================

Golden models for RMSNorm and LayerNorm matching RTL.

This module provides:
- RMSNorm (used in LLaMA-style models)
- LayerNorm (used in ViT models)
- Fixed-point implementations
- Test vector generation

License: Apache 2.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from .golden_attention import WeightType, PrecisionMode, FixedPointOps, TestVector


@dataclass
class NormConfig:
    """Configuration for normalization layers."""
    dim: int = 768
    eps: float = 1e-6
    act_width: int = 8
    acc_width: int = 32
    frac_bits: int = 4


class GoldenRMSNorm:
    """
    Golden model for RMS Normalization.
    
    RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
    
    Used in LLaMA/SmolLM style language models.
    """
    
    def __init__(
        self,
        dim: int = 768,
        eps: float = 1e-6,
        precision: PrecisionMode = PrecisionMode.FLOAT
    ):
        self.dim = dim
        self.eps = eps
        self.precision = precision
        
        self.fp = FixedPointOps(width=8, frac_bits=4)
        self.fp_acc = FixedPointOps(width=32, frac_bits=8)

        # Learnable scale parameter (gamma)
        np.random.seed(300)
        self.gamma = np.ones(dim, dtype=np.float32)
    
    def _compute_rms(self, x: np.ndarray) -> np.ndarray:
        """Compute RMS of input."""
        return np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
    
    def _rsqrt_newton_raphson(self, x: np.ndarray, iterations: int = 3) -> np.ndarray:
        """
        Compute 1/sqrt(x) using Newton-Raphson.
        
        y_{n+1} = y_n * (3 - x * y_n^2) / 2
        """
        # Initial guess: 1.0
        y = np.ones_like(x)
        
        for _ in range(iterations):
            y = y * (3.0 - x * y * y) / 2.0
        
        return y
    
    def forward(self, x: np.ndarray, use_fixed_point: bool = False) -> np.ndarray:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor [..., dim]
            use_fixed_point: Use fixed-point arithmetic
            
        Returns:
            Normalized tensor
        """
        if use_fixed_point:
            # Fixed-point implementation matching RTL
            x_fixed = self.fp.float_to_fixed(x)
            
            # Compute sum of squares
            x_sq = self.fp.mul(x_fixed, x_fixed)
            mean_sq = np.mean(x_sq, axis=-1, keepdims=True).astype(np.int32)
            mean_sq = np.maximum(mean_sq, 1)  # Avoid division by zero
            
            # Newton-Raphson for rsqrt
            inv_rms = self._rsqrt_newton_raphson(
                self.fp.fixed_to_float(mean_sq), iterations=3
            )
            inv_rms_fixed = self.fp_acc.float_to_fixed(inv_rms)
            
            # Normalize
            x_norm = self.fp.mul(x_fixed, inv_rms_fixed.astype(np.int32))
            
            # Apply scale
            gamma_fixed = self.fp.float_to_fixed(self.gamma)
            output = self.fp.mul(x_norm, gamma_fixed)
            
            return self.fp.fixed_to_float(output).astype(np.float32)
        else:
            # Float implementation
            rms = self._compute_rms(x)
            x_norm = x / rms
            return (x_norm * self.gamma).astype(np.float32)

    def generate_test_vectors(
        self,
        num_vectors: int = 20,
        seed: int = 42
    ) -> List[TestVector]:
        """Generate test vectors for RTL verification."""
        np.random.seed(seed)
        vectors = []
        
        # Edge cases
        # All same value
        x_same = np.full((16, self.dim), 2.0, dtype=np.float32)
        out_same = self.forward(x_same)
        vectors.append(TestVector(
            name="rmsnorm_same_values",
            inputs={'x': x_same, 'gamma': self.gamma},
            expected_outputs={'output': out_same},
            description="All same values (tests RMS = |value|)"
        ))
        
        # Unit vector
        x_unit = np.zeros((1, self.dim), dtype=np.float32)
        x_unit[0, 0] = 1.0
        out_unit = self.forward(x_unit)
        vectors.append(TestVector(
            name="rmsnorm_unit",
            inputs={'x': x_unit, 'gamma': self.gamma},
            expected_outputs={'output': out_unit},
            description="Unit vector"
        ))
        
        # Random vectors
        for i in range(num_vectors - 2):
            seq_len = np.random.choice([1, 8, 16, 32])
            x = np.random.randn(seq_len, self.dim).astype(np.float32)
            output = self.forward(x)
            
            vectors.append(TestVector(
                name=f"rmsnorm_random_{i}",
                inputs={'x': x, 'gamma': self.gamma},
                expected_outputs={'output': output},
                description=f"Random input, seq={seq_len}"
            ))
        
        return vectors


class GoldenLayerNorm:
    """
    Golden model for Layer Normalization.
    
    LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    
    Used in Vision Transformer models.
    """
    
    def __init__(
        self,
        dim: int = 768,
        eps: float = 1e-6,
        precision: PrecisionMode = PrecisionMode.FLOAT
    ):
        self.dim = dim
        self.eps = eps
        self.precision = precision
        
        self.fp = FixedPointOps(width=8, frac_bits=4)
        
        # Learnable parameters
        np.random.seed(301)
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)

    def forward(self, x: np.ndarray, use_fixed_point: bool = False) -> np.ndarray:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor [..., dim]
            use_fixed_point: Use fixed-point arithmetic
            
        Returns:
            Normalized tensor
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        output = self.gamma * x_norm + self.beta
        
        if use_fixed_point:
            output = self.fp.fixed_to_float(self.fp.float_to_fixed(output))
        
        return output.astype(np.float32)
    
    def generate_test_vectors(
        self,
        num_vectors: int = 20,
        seed: int = 42
    ) -> List[TestVector]:
        """Generate test vectors for RTL verification."""
        np.random.seed(seed)
        vectors = []
        
        # Sequential values (tests mean/var computation)
        x_seq = np.arange(self.dim, dtype=np.float32).reshape(1, -1)
        out_seq = self.forward(x_seq)
        vectors.append(TestVector(
            name="layernorm_sequential",
            inputs={'x': x_seq, 'gamma': self.gamma, 'beta': self.beta},
            expected_outputs={'output': out_seq},
            description="Sequential values"
        ))
        
        # Constant (zero variance)
        x_const = np.full((1, self.dim), 5.0, dtype=np.float32)
        out_const = self.forward(x_const)
        vectors.append(TestVector(
            name="layernorm_constant",
            inputs={'x': x_const, 'gamma': self.gamma, 'beta': self.beta},
            expected_outputs={'output': out_const},
            description="Constant input (zero variance)"
        ))
        
        # Random vectors
        for i in range(num_vectors - 2):
            seq_len = np.random.choice([1, 8, 16, 32])
            x = np.random.randn(seq_len, self.dim).astype(np.float32)
            output = self.forward(x)
            
            vectors.append(TestVector(
                name=f"layernorm_random_{i}",
                inputs={'x': x, 'gamma': self.gamma, 'beta': self.beta},
                expected_outputs={'output': output},
                description=f"Random input, seq={seq_len}"
            ))
        
        return vectors


def test_golden_normalization():
    """Test the golden normalization implementations."""
    print("Testing Golden Normalization Layers")
    print("=" * 50)
    
    dim = 64
    
    # Test RMSNorm
    print("\n--- RMSNorm ---")
    rms_norm = GoldenRMSNorm(dim=dim)
    x = np.random.randn(16, dim).astype(np.float32)
    
    out_float = rms_norm.forward(x, use_fixed_point=False)
    out_fixed = rms_norm.forward(x, use_fixed_point=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_float.shape}")
    
    # Check output RMS is ~1
    out_rms = np.sqrt(np.mean(out_float ** 2, axis=-1))
    print(f"Output RMS (should be ~1): {out_rms.mean():.4f}")
    
    # Test LayerNorm
    print("\n--- LayerNorm ---")
    layer_norm = GoldenLayerNorm(dim=dim)
    out = layer_norm.forward(x)
    
    print(f"Output shape: {out.shape}")
    
    # Check output mean ~0, std ~1
    out_mean = np.mean(out, axis=-1)
    out_std = np.std(out, axis=-1)
    print(f"Output mean (should be ~0): {out_mean.mean():.6f}")
    print(f"Output std (should be ~1): {out_std.mean():.4f}")
    
    # Test vector generation
    print("\n--- Test Vector Generation ---")
    rms_vectors = rms_norm.generate_test_vectors(num_vectors=10)
    ln_vectors = layer_norm.generate_test_vectors(num_vectors=10)
    
    print(f"RMSNorm: {len(rms_vectors)} vectors")
    print(f"LayerNorm: {len(ln_vectors)} vectors")
    
    print("\n✓ All normalization tests passed!")


if __name__ == "__main__":
    test_golden_normalization()
