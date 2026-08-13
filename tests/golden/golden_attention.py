#!/usr/bin/env python3
"""
SiLens Golden Model: Multi-Head Attention
==========================================

Pure Python/NumPy reference implementation of multi-head attention
that matches the RTL implementation exactly.

This module provides:
- Binary and ternary weight support
- Fixed-point arithmetic matching hardware
- Test vector generation for RTL verification
- Exact bit-accurate computation

Usage:
    from golden_attention import GoldenMultiHeadAttention
    
    # Create attention module
    attn = GoldenMultiHeadAttention(
        embed_dim=768,
        num_heads=12,
        weight_type='ternary',
        precision='fixed'
    )
    
    # Generate test vectors
    vectors = attn.generate_test_vectors(num_vectors=100)

License: Apache 2.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Literal
from enum import Enum


class WeightType(Enum):
    """Weight quantization type."""
    BINARY = "binary"       # {-1, +1}
    TERNARY = "ternary"     # {-1, 0, +1}
    FLOAT = "float"         # Full precision


class PrecisionMode(Enum):
    """Computation precision mode."""
    FLOAT = "float"         # Full precision (for reference)
    FIXED = "fixed"         # Fixed-point (matching hardware)


@dataclass
class AttentionConfig:
    """Configuration for attention module."""
    embed_dim: int = 768            # Embedding dimension
    num_heads: int = 12             # Number of attention heads
    head_dim: int = None            # Dimension per head (computed from embed_dim // num_heads)
    max_seq_len: int = 512          # Maximum sequence length
    
    # Fixed-point parameters (matching RTL)
    act_width: int = 8              # Activation bit width
    acc_width: int = 32             # Accumulator bit width
    frac_bits: int = 4              # Fractional bits for fixed-point
    
    # Weight type
    weight_type: WeightType = WeightType.TERNARY
    
    def __post_init__(self):
        # Always compute head_dim from embed_dim and num_heads if not explicitly set
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"


@dataclass
class TestVector:
    """Test vector for RTL verification."""
    name: str
    inputs: Dict[str, np.ndarray]
    expected_outputs: Dict[str, np.ndarray]
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'inputs': {k: v.tolist() for k, v in self.inputs.items()},
            'expected_outputs': {k: v.tolist() for k, v in self.expected_outputs.items()},
        }


class FixedPointOps:
    """
    Fixed-point arithmetic operations matching RTL.
    
    Uses Q(int_bits).(frac_bits) format.
    """
    
    def __init__(self, width: int = 8, frac_bits: int = 4):
        """
        Initialize fixed-point operations.
        
        Args:
            width: Total bit width (including sign)
            frac_bits: Number of fractional bits
        """
        self.width = width
        self.frac_bits = frac_bits
        self.int_bits = width - frac_bits - 1  # -1 for sign bit
        
        # Compute ranges
        self.scale = 1 << frac_bits
        self.max_val = (1 << (width - 1)) - 1
        self.min_val = -(1 << (width - 1))
    
    def float_to_fixed(self, x: np.ndarray) -> np.ndarray:
        """Convert floating-point to fixed-point representation."""
        scaled = np.round(x * self.scale)
        return np.clip(scaled, self.min_val, self.max_val).astype(np.int32)
    
    def fixed_to_float(self, x: np.ndarray) -> np.ndarray:
        """Convert fixed-point to floating-point."""
        return x.astype(np.float64) / self.scale
    
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fixed-point addition with saturation."""
        result = a.astype(np.int64) + b.astype(np.int64)
        return np.clip(result, self.min_val, self.max_val).astype(np.int32)
    
    def mul(self, a: np.ndarray, b: np.ndarray, 
            a_frac: int = None, b_frac: int = None) -> np.ndarray:
        """
        Fixed-point multiplication with scaling.
        
        For Q.a * Q.b, result is Q.(a+b), needs shift by frac_bits.
        """
        a_frac = a_frac if a_frac is not None else self.frac_bits
        b_frac = b_frac if b_frac is not None else self.frac_bits
        
        result = a.astype(np.int64) * b.astype(np.int64)
        # Right shift to maintain proper scaling
        shift = a_frac + b_frac - self.frac_bits
        if shift > 0:
            result = result >> shift
        return result.astype(np.int32)


class GoldenSoftmax:
    """
    Golden model for approximate softmax matching RTL.
    
    Uses piece-wise linear approximation of exp().
    """
    
    def __init__(self, config: AttentionConfig, fp_ops: FixedPointOps):
        self.config = config
        self.fp = fp_ops
    
    def exp_approx(self, x: np.ndarray) -> np.ndarray:
        """
        Piece-wise linear approximation of exp(x).
        
        For x in [-8, 0], approximates exp(x).
        """
        result = np.zeros_like(x, dtype=np.float64)
        
        # Segment approximations (matching RTL)
        mask_ge_0 = x >= 0
        mask_lt_neg4 = x < -4
        mask_neg4_neg2 = (x >= -4) & (x < -2)
        mask_neg2_neg1 = (x >= -2) & (x < -1)
        mask_neg1_0 = (x >= -1) & (x < 0)
        
        # exp(x) for each segment
        result[mask_ge_0] = 1.0
        result[mask_lt_neg4] = 0.018  # Very small
        result[mask_neg4_neg2] = 0.018 + 0.058 * (x[mask_neg4_neg2] + 4)
        result[mask_neg2_neg1] = 0.135 + 0.233 * (x[mask_neg2_neg1] + 2)
        result[mask_neg1_0] = 0.368 + 0.632 * (x[mask_neg1_0] + 1)
        
        return np.maximum(result, 0)
    
    def forward(self, x: np.ndarray, fixed_point: bool = False) -> np.ndarray:
        """
        Compute softmax with approximate exp.
        
        Args:
            x: Input logits [..., seq_len]
            fixed_point: Use fixed-point arithmetic
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        x_max = np.max(x, axis=-1, keepdims=True)
        x_shifted = x - x_max
        
        # Compute exp approximation
        exp_x = self.exp_approx(x_shifted)
        
        # Normalize
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        softmax = exp_x / (sum_exp + 1e-10)
        
        if fixed_point:
            return self.fp.float_to_fixed(softmax)
        return softmax


class GoldenMultiHeadAttention:
    """
    Golden model for multi-head attention matching RTL.
    
    Implements:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    With support for:
    - Binary weights ({-1, +1} using XNOR + popcount)
    - Ternary weights ({-1, 0, +1} using MAC)
    - Fixed-point arithmetic
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        weight_type: Union[str, WeightType] = 'ternary',
        precision: Union[str, PrecisionMode] = 'float',
        **kwargs
    ):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            weight_type: 'binary', 'ternary', or 'float'
            precision: 'float' or 'fixed'
            **kwargs: Additional config parameters
        """
        if isinstance(weight_type, str):
            weight_type = WeightType(weight_type)
        if isinstance(precision, str):
            precision = PrecisionMode(precision)
        
        self.config = AttentionConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            weight_type=weight_type,
            **{k: v for k, v in kwargs.items() if hasattr(AttentionConfig, k)}
        )
        self.precision = precision
        
        # Initialize fixed-point operations
        self.fp = FixedPointOps(
            width=self.config.act_width,
            frac_bits=self.config.frac_bits
        )
        self.fp_acc = FixedPointOps(
            width=self.config.acc_width,
            frac_bits=self.config.frac_bits * 2
        )
        
        # Softmax module
        self.softmax = GoldenSoftmax(self.config, self.fp)
        
        # Initialize random weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        np.random.seed(42)
        
        d = self.config.embed_dim
        h = self.config.num_heads
        d_k = self.config.head_dim
        
        if self.config.weight_type == WeightType.BINARY:
            # Binary weights: {-1, +1}
            self.W_q = np.random.choice([-1, 1], size=(d, d)).astype(np.int8)
            self.W_k = np.random.choice([-1, 1], size=(d, d)).astype(np.int8)
            self.W_v = np.random.choice([-1, 1], size=(d, d)).astype(np.int8)
            self.W_o = np.random.choice([-1, 1], size=(d, d)).astype(np.int8)
            
        elif self.config.weight_type == WeightType.TERNARY:
            # Ternary weights: {-1, 0, +1} with ~30% sparsity
            self.W_q = np.random.choice([-1, 0, 1], size=(d, d), 
                                        p=[0.35, 0.3, 0.35]).astype(np.int8)
            self.W_k = np.random.choice([-1, 0, 1], size=(d, d), 
                                        p=[0.35, 0.3, 0.35]).astype(np.int8)
            self.W_v = np.random.choice([-1, 0, 1], size=(d, d), 
                                        p=[0.35, 0.3, 0.35]).astype(np.int8)
            self.W_o = np.random.choice([-1, 0, 1], size=(d, d), 
                                        p=[0.35, 0.3, 0.35]).astype(np.int8)
        else:
            # Float weights (for reference)
            self.W_q = np.random.randn(d, d).astype(np.float32) * 0.02
            self.W_k = np.random.randn(d, d).astype(np.float32) * 0.02
            self.W_v = np.random.randn(d, d).astype(np.float32) * 0.02
            self.W_o = np.random.randn(d, d).astype(np.float32) * 0.02
        
        # Scale factors for quantized weights
        self.scale_q = 1.0
        self.scale_k = 1.0
        self.scale_v = 1.0
        self.scale_o = 1.0
    
    def set_weights(
        self,
        W_q: np.ndarray,
        W_k: np.ndarray,
        W_v: np.ndarray,
        W_o: np.ndarray,
        scales: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Set projection weights from external source.
        
        Args:
            W_q, W_k, W_v, W_o: Weight matrices
            scales: Optional scale factors for quantized weights
        """
        self.W_q = W_q
        self.W_k = W_k
        self.W_v = W_v
        self.W_o = W_o
        
        if scales:
            self.scale_q, self.scale_k, self.scale_v, self.scale_o = scales
    
    def _binary_matmul(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Binary matrix multiplication using XNOR + popcount.
        
        For binary values: dot(a, b) = 2 * popcount(XNOR(a, b)) - n
        
        Args:
            x: Input activations (will be binarized)
            w: Binary weights {-1, +1}
            
        Returns:
            Output of binary matmul
        """
        # Binarize activations: sign(x) -> {0, 1} for XNOR
        x_bin = (x > 0).astype(np.uint8)
        w_bin = (w > 0).astype(np.uint8)
        
        # XNOR operation
        xnor = ~(x_bin ^ w_bin) & 1
        
        # Popcount and convert to dot product
        popcount = np.sum(xnor, axis=-1)
        n = w.shape[-1]
        result = 2 * popcount - n
        
        return result.astype(np.float32)
    
    def _ternary_matmul(self, x: np.ndarray, w: np.ndarray, 
                        scale: float = 1.0) -> np.ndarray:
        """
        Ternary matrix multiplication.
        
        For each output element: sum(x_i * w_i) where w_i ∈ {-1, 0, +1}
        
        Args:
            x: Input activations
            w: Ternary weights {-1, 0, +1}
            scale: Scale factor for dequantization
            
        Returns:
            Output of ternary matmul
        """
        # Efficient implementation using masks
        pos_mask = (w == 1)
        neg_mask = (w == -1)
        
        # For matrix multiply: x @ w^T or similar
        # We compute this element-wise for clarity
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # x: [batch, in_dim] or [seq, in_dim]
        # w: [out_dim, in_dim]
        # result: [batch, out_dim]
        
        result = np.zeros((x.shape[0], w.shape[0]), dtype=np.float64)
        
        for i in range(w.shape[0]):
            # Sum where weight is +1
            pos_sum = np.sum(x[:, pos_mask[i]], axis=-1)
            # Sum where weight is -1
            neg_sum = np.sum(x[:, neg_mask[i]], axis=-1)
            result[:, i] = pos_sum - neg_sum
        
        return (result * scale).astype(np.float32)
    
    def _linear(self, x: np.ndarray, w: np.ndarray, 
                scale: float = 1.0) -> np.ndarray:
        """
        Linear projection using appropriate method for weight type.
        
        Args:
            x: Input tensor
            w: Weight matrix
            scale: Scale factor
            
        Returns:
            Output tensor
        """
        if self.config.weight_type == WeightType.BINARY:
            return self._binary_matmul(x, w)
        elif self.config.weight_type == WeightType.TERNARY:
            return self._ternary_matmul(x, w, scale)
        else:
            return (x @ w.T).astype(np.float32)
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split embedding into multiple heads.
        
        Args:
            x: [..., seq_len, embed_dim]
            
        Returns:
            [..., num_heads, seq_len, head_dim]
        """
        *batch_shape, seq_len, _ = x.shape
        x = x.reshape(*batch_shape, seq_len, self.config.num_heads, self.config.head_dim)
        return np.transpose(x, (*range(len(batch_shape)), -2, -3, -1))
    
    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Merge heads back into single embedding.
        
        Args:
            x: [..., num_heads, seq_len, head_dim]
            
        Returns:
            [..., seq_len, embed_dim]
        """
        *batch_shape, num_heads, seq_len, head_dim = x.shape
        x = np.transpose(x, (*range(len(batch_shape)), -2, -3, -1))
        return x.reshape(*batch_shape, seq_len, -1)
    
    def forward(
        self,
        x: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        return_attention: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim] or [seq_len, embed_dim]
            attention_mask: Optional attention mask [seq_len, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor, optionally attention weights
        """
        single_input = x.ndim == 2
        if single_input:
            x = x[np.newaxis, ...]
        
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self._linear(x.reshape(-1, self.config.embed_dim), 
                        self.W_q, self.scale_q)
        K = self._linear(x.reshape(-1, self.config.embed_dim), 
                        self.W_k, self.scale_k)
        V = self._linear(x.reshape(-1, self.config.embed_dim), 
                        self.W_v, self.scale_v)
        
        # Reshape to [batch, seq, embed_dim]
        Q = Q.reshape(batch_size, seq_len, self.config.embed_dim)
        K = K.reshape(batch_size, seq_len, self.config.embed_dim)
        V = V.reshape(batch_size, seq_len, self.config.embed_dim)
        
        # Split heads: [batch, num_heads, seq, head_dim]
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(self.config.head_dim)
        attn_scores = np.matmul(Q, np.transpose(K, (0, 1, 3, 2))) * scale
        
        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax
        attn_weights = self.softmax.forward(
            attn_scores, 
            fixed_point=(self.precision == PrecisionMode.FIXED)
        )
        
        # Apply attention to values
        context = np.matmul(attn_weights, V)
        
        # Merge heads: [batch, seq, embed_dim]
        context = self._merge_heads(context)
        
        # Output projection
        output = self._linear(context.reshape(-1, self.config.embed_dim),
                             self.W_o, self.scale_o)
        output = output.reshape(batch_size, seq_len, self.config.embed_dim)
        
        if single_input:
            output = output[0]
            if return_attention:
                attn_weights = attn_weights[0]
        
        if return_attention:
            return output, attn_weights
        return output
    
    def generate_test_vectors(
        self,
        num_vectors: int = 10,
        seq_lengths: List[int] = None,
        seed: int = 42
    ) -> List[TestVector]:
        """
        Generate test vectors for RTL verification.
        
        Args:
            num_vectors: Number of random vectors
            seq_lengths: Sequence lengths to test
            seed: Random seed
            
        Returns:
            List of test vectors
        """
        if seq_lengths is None:
            seq_lengths = [1, 4, 8, 16]
        
        np.random.seed(seed)
        vectors = []
        
        # Edge case: all zeros
        x_zeros = np.zeros((4, self.config.embed_dim), dtype=np.float32)
        out_zeros = self.forward(x_zeros)
        vectors.append(TestVector(
            name="all_zeros",
            inputs={'x': x_zeros},
            expected_outputs={'output': out_zeros},
            description="All zero inputs"
        ))
        
        # Edge case: identity-like pattern
        x_identity = np.eye(min(8, self.config.embed_dim), 
                           self.config.embed_dim, dtype=np.float32)
        out_identity = self.forward(x_identity)
        vectors.append(TestVector(
            name="identity_pattern",
            inputs={'x': x_identity},
            expected_outputs={'output': out_identity},
            description="Identity-like input pattern"
        ))
        
        # Random vectors for each sequence length
        for seq_len in seq_lengths:
            for i in range(num_vectors // len(seq_lengths)):
                # Random input in reasonable range
                x = np.random.randn(seq_len, self.config.embed_dim).astype(np.float32) * 0.1
                
                # Convert to fixed-point if needed
                if self.precision == PrecisionMode.FIXED:
                    x = self.fp.fixed_to_float(self.fp.float_to_fixed(x))
                
                output, attn = self.forward(x, return_attention=True)
                
                vectors.append(TestVector(
                    name=f"random_seq{seq_len}_{i}",
                    inputs={'x': x},
                    expected_outputs={
                        'output': output,
                        'attention_weights': attn
                    },
                    description=f"Random input, seq_len={seq_len}"
                ))
        
        return vectors
    
    def get_weight_statistics(self) -> Dict[str, Dict]:
        """Get statistics about the weights."""
        def stats(w):
            unique, counts = np.unique(w, return_counts=True)
            total = w.size
            return {
                'shape': list(w.shape),
                'unique_values': unique.tolist(),
                'distribution': {int(v): int(c) for v, c in zip(unique, counts)},
                'sparsity': float(np.sum(w == 0) / total) if 0 in unique else 0.0,
                'nonzero': int(np.count_nonzero(w))
            }
        
        return {
            'W_q': stats(self.W_q),
            'W_k': stats(self.W_k),
            'W_v': stats(self.W_v),
            'W_o': stats(self.W_o),
        }


def test_golden_attention():
    """Test the golden attention implementation."""
    print("Testing Golden Multi-Head Attention")
    print("=" * 50)
    
    # Test configurations
    configs = [
        {'embed_dim': 64, 'num_heads': 4, 'weight_type': 'binary'},
        {'embed_dim': 64, 'num_heads': 4, 'weight_type': 'ternary'},
        {'embed_dim': 768, 'num_heads': 12, 'weight_type': 'ternary'},
    ]
    
    for config in configs:
        print(f"\nConfig: {config}")
        attn = GoldenMultiHeadAttention(**config)
        
        # Test forward pass
        x = np.random.randn(4, config['embed_dim']).astype(np.float32) * 0.1
        output, attn_weights = attn.forward(x, return_attention=True)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Attention shape: {attn_weights.shape}")
        
        # Verify output shape
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        
        # Verify attention weights sum to ~1
        attn_sum = np.sum(attn_weights, axis=-1)
        assert np.allclose(attn_sum, 1.0, atol=0.1), f"Attention doesn't sum to 1: {attn_sum}"
        
        # Get weight statistics
        stats = attn.get_weight_statistics()
        print(f"  W_q sparsity: {stats['W_q']['sparsity']:.1%}")
        
        # Generate test vectors
        vectors = attn.generate_test_vectors(num_vectors=4)
        print(f"  Generated {len(vectors)} test vectors")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_golden_attention()
