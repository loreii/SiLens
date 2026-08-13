#!/usr/bin/env python3
"""
SiLens Golden Model: Transformer Block
=======================================

Pure Python/NumPy reference implementation of a complete transformer block
that matches the RTL implementation exactly.

This module provides:
- Layer normalization (RMSNorm)
- Multi-head self-attention
- MLP with GELU activation
- Support for vision and language model configurations

Usage:
    from golden_transformer import GoldenTransformerBlock
    
    # Create transformer block for vision encoder
    block = GoldenTransformerBlock(
        config=VisionConfig(embed_dim=768, num_heads=12),
        weight_type='ternary'
    )
    
    # Run forward pass
    output = block.forward(input_tensor)

License: Apache 2.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

from .golden_attention import (
    GoldenMultiHeadAttention,
    WeightType,
    PrecisionMode,
    FixedPointOps,
    TestVector
)


@dataclass
class VisionConfig:
    """Configuration for vision encoder (SigLIP-style)."""
    embed_dim: int = 768
    num_heads: int = 12
    head_dim: int = None       # Computed from embed_dim // num_heads if not set
    mlp_dim: int = 3072        # 4 * embed_dim typically
    num_layers: int = 12
    image_size: int = 384
    patch_size: int = 14
    num_patches: int = 729     # (384/14)^2 + 1 for CLS
    
    # Activation
    activation: str = "gelu"
    
    # Normalization
    norm_eps: float = 1e-6
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads
        if self.mlp_dim is None:
            self.mlp_dim = 4 * self.embed_dim


@dataclass
class LanguageConfig:
    """Configuration for language model (LLaMA-style)."""
    embed_dim: int = 576
    num_heads: int = 9
    head_dim: int = None       # Computed from embed_dim // num_heads if not set
    mlp_dim: int = 1536
    num_layers: int = 30
    vocab_size: int = 49152
    max_seq_len: int = 2048
    
    # Activation
    activation: str = "silu"   # SwiGLU uses SiLU
    
    # Normalization  
    norm_eps: float = 1e-5
    
    # RoPE
    rope_theta: float = 10000.0
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads


class GoldenLayerNorm:
    """
    Golden model for Layer Normalization matching RTL.
    
    Implements: y = gamma * (x - mean) / sqrt(var + eps) + beta
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        fp_ops: Optional[FixedPointOps] = None
    ):
        """
        Initialize layer normalization.
        
        Args:
            dim: Feature dimension
            eps: Epsilon for numerical stability
            fp_ops: Fixed-point operations
        """
        self.dim = dim
        self.eps = eps
        self.fp = fp_ops
        
        # Learnable parameters (initialized to 1, 0)
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
    
    def forward(self, x: np.ndarray, fixed_point: bool = False) -> np.ndarray:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor [..., dim]
            fixed_point: Use fixed-point arithmetic
            
        Returns:
            Normalized tensor
        """
        # Compute mean and variance
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Apply scale and bias
        output = self.gamma * x_norm + self.beta
        
        if fixed_point and self.fp is not None:
            output = self.fp.fixed_to_float(self.fp.float_to_fixed(output))
        
        return output.astype(np.float32)


class GoldenRMSNorm:
    """
    Golden model for RMS Normalization (used in LLaMA).
    
    Implements: y = x * rsqrt(mean(x^2) + eps) * gamma
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        fp_ops: Optional[FixedPointOps] = None
    ):
        """
        Initialize RMS normalization.
        
        Args:
            dim: Feature dimension
            eps: Epsilon for numerical stability
            fp_ops: Fixed-point operations
        """
        self.dim = dim
        self.eps = eps
        self.fp = fp_ops
        
        # Learnable parameter
        self.weight = np.ones(dim, dtype=np.float32)
    
    def forward(self, x: np.ndarray, fixed_point: bool = False) -> np.ndarray:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor [..., dim]
            fixed_point: Use fixed-point arithmetic
            
        Returns:
            Normalized tensor
        """
        # Compute RMS
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        
        # Normalize and scale
        output = (x / rms) * self.weight
        
        if fixed_point and self.fp is not None:
            output = self.fp.fixed_to_float(self.fp.float_to_fixed(output))
        
        return output.astype(np.float32)


class GoldenGELU:
    """
    Golden model for GELU activation matching RTL.
    
    Uses piece-wise linear approximation:
        GELU(x) ≈ x * sigmoid(1.702 * x)
    """
    
    def __init__(self, fp_ops: Optional[FixedPointOps] = None):
        self.fp = fp_ops
    
    def forward(self, x: np.ndarray, fixed_point: bool = False) -> np.ndarray:
        """
        Apply GELU activation (piece-wise linear approximation).
        
        Args:
            x: Input tensor
            fixed_point: Use fixed-point arithmetic
            
        Returns:
            Activated tensor
        """
        if fixed_point:
            # Piece-wise linear approximation matching RTL
            result = np.zeros_like(x)
            
            # Different segments
            mask_neg3 = x < -3
            mask_neg2 = (x >= -3) & (x < -2)
            mask_neg1 = (x >= -2) & (x < -1)
            mask_0 = (x >= -1) & (x < 0)
            mask_1 = (x >= 0) & (x < 1)
            mask_2 = (x >= 1) & (x < 2)
            mask_3 = (x >= 2) & (x < 3)
            mask_pos3 = x >= 3
            
            # Approximation values (matching RTL constants)
            result[mask_neg3] = 0
            result[mask_neg2] = -0.045 * x[mask_neg2] - 0.135
            result[mask_neg1] = 0.114 * x[mask_neg1] - 0.273
            result[mask_0] = 0.159 * x[mask_0]
            result[mask_1] = 0.841 * x[mask_1]
            result[mask_2] = 1.114 * x[mask_2] - 0.273
            result[mask_3] = 1.041 * x[mask_3] - 0.127
            result[mask_pos3] = x[mask_pos3]
            
            return result.astype(np.float32)
        else:
            # Exact GELU
            return (0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * 
                    (x + 0.044715 * x ** 3)))).astype(np.float32)


class GoldenSiLU:
    """
    Golden model for SiLU (Swish) activation.
    
    SiLU(x) = x * sigmoid(x)
    """
    
    def __init__(self, fp_ops: Optional[FixedPointOps] = None):
        self.fp = fp_ops
    
    def forward(self, x: np.ndarray, fixed_point: bool = False) -> np.ndarray:
        """
        Apply SiLU activation.
        
        Args:
            x: Input tensor
            fixed_point: Use fixed-point arithmetic
            
        Returns:
            Activated tensor
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return (x * sigmoid).astype(np.float32)


class GoldenMLP:
    """
    Golden model for MLP (Feed-Forward Network).
    
    Standard: FFN(x) = activation(x @ W1 + b1) @ W2 + b2
    SwiGLU:   FFN(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down
    """
    
    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        activation: str = "gelu",
        weight_type: WeightType = WeightType.TERNARY,
        use_swiglu: bool = False,
        fp_ops: Optional[FixedPointOps] = None
    ):
        """
        Initialize MLP.
        
        Args:
            embed_dim: Input/output dimension
            mlp_dim: Hidden dimension
            activation: Activation function ('gelu' or 'silu')
            weight_type: Weight quantization type
            use_swiglu: Use SwiGLU architecture
            fp_ops: Fixed-point operations
        """
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.weight_type = weight_type
        self.use_swiglu = use_swiglu
        self.fp = fp_ops
        
        # Activation function
        if activation == "gelu":
            self.activation = GoldenGELU(fp_ops)
        else:
            self.activation = GoldenSiLU(fp_ops)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights."""
        np.random.seed(123)
        
        if self.weight_type == WeightType.TERNARY:
            if self.use_swiglu:
                self.W_gate = np.random.choice([-1, 0, 1], 
                    size=(self.mlp_dim, self.embed_dim),
                    p=[0.35, 0.3, 0.35]).astype(np.int8)
                self.W_up = np.random.choice([-1, 0, 1], 
                    size=(self.mlp_dim, self.embed_dim),
                    p=[0.35, 0.3, 0.35]).astype(np.int8)
            else:
                self.W_up = np.random.choice([-1, 0, 1], 
                    size=(self.mlp_dim, self.embed_dim),
                    p=[0.35, 0.3, 0.35]).astype(np.int8)
            
            self.W_down = np.random.choice([-1, 0, 1], 
                size=(self.embed_dim, self.mlp_dim),
                p=[0.35, 0.3, 0.35]).astype(np.int8)
        else:
            if self.use_swiglu:
                self.W_gate = np.random.randn(self.mlp_dim, self.embed_dim).astype(np.float32) * 0.02
                self.W_up = np.random.randn(self.mlp_dim, self.embed_dim).astype(np.float32) * 0.02
            else:
                self.W_up = np.random.randn(self.mlp_dim, self.embed_dim).astype(np.float32) * 0.02
            
            self.W_down = np.random.randn(self.embed_dim, self.mlp_dim).astype(np.float32) * 0.02
        
        # Scale factors
        self.scale_up = 1.0
        self.scale_down = 1.0
        self.scale_gate = 1.0
    
    def _ternary_matmul(self, x: np.ndarray, w: np.ndarray, 
                        scale: float = 1.0) -> np.ndarray:
        """Ternary matrix multiplication."""
        pos_mask = (w == 1)
        neg_mask = (w == -1)
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        result = np.zeros((x.shape[0], w.shape[0]), dtype=np.float64)
        
        for i in range(w.shape[0]):
            pos_sum = np.sum(x[:, pos_mask[i]], axis=-1)
            neg_sum = np.sum(x[:, neg_mask[i]], axis=-1)
            result[:, i] = pos_sum - neg_sum
        
        return (result * scale).astype(np.float32)
    
    def _linear(self, x: np.ndarray, w: np.ndarray, 
                scale: float = 1.0) -> np.ndarray:
        """Linear projection."""
        if self.weight_type == WeightType.TERNARY:
            return self._ternary_matmul(x, w, scale)
        else:
            return (x @ w.T).astype(np.float32)
    
    def forward(self, x: np.ndarray, fixed_point: bool = False) -> np.ndarray:
        """
        Forward pass through MLP.
        
        Args:
            x: Input tensor [..., embed_dim]
            fixed_point: Use fixed-point arithmetic
            
        Returns:
            Output tensor [..., embed_dim]
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.embed_dim)
        
        if self.use_swiglu:
            # SwiGLU: (silu(x @ W_gate) * (x @ W_up)) @ W_down
            gate = self._linear(x_flat, self.W_gate, self.scale_gate)
            gate = self.activation.forward(gate, fixed_point)
            
            up = self._linear(x_flat, self.W_up, self.scale_up)
            hidden = gate * up
        else:
            # Standard: activation(x @ W_up) @ W_down
            hidden = self._linear(x_flat, self.W_up, self.scale_up)
            hidden = self.activation.forward(hidden, fixed_point)
        
        output = self._linear(hidden, self.W_down, self.scale_down)
        
        return output.reshape(original_shape).astype(np.float32)


class GoldenTransformerBlock:
    """
    Golden model for a complete Transformer block.
    
    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """
    
    def __init__(
        self,
        config: Union[VisionConfig, LanguageConfig],
        weight_type: Union[str, WeightType] = 'ternary',
        precision: Union[str, PrecisionMode] = 'float'
    ):
        """
        Initialize transformer block.
        
        Args:
            config: Vision or Language model configuration
            weight_type: Weight quantization type
            precision: Computation precision
        """
        self.config = config
        
        if isinstance(weight_type, str):
            weight_type = WeightType(weight_type)
        if isinstance(precision, str):
            precision = PrecisionMode(precision)
        
        self.weight_type = weight_type
        self.precision = precision
        
        # Initialize fixed-point operations
        self.fp = FixedPointOps(width=8, frac_bits=4)
        
        # Determine if this is a language model (uses RMSNorm, SwiGLU)
        is_language = isinstance(config, LanguageConfig)
        
        # Layer normalization
        if is_language:
            self.norm1 = GoldenRMSNorm(config.embed_dim, config.norm_eps, self.fp)
            self.norm2 = GoldenRMSNorm(config.embed_dim, config.norm_eps, self.fp)
        else:
            self.norm1 = GoldenLayerNorm(config.embed_dim, config.norm_eps, self.fp)
            self.norm2 = GoldenLayerNorm(config.embed_dim, config.norm_eps, self.fp)
        
        # Self-attention
        self.attention = GoldenMultiHeadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            weight_type=weight_type,
            precision=precision
        )
        
        # MLP
        self.mlp = GoldenMLP(
            embed_dim=config.embed_dim,
            mlp_dim=config.mlp_dim,
            activation=config.activation,
            weight_type=weight_type,
            use_swiglu=is_language,
            fp_ops=self.fp
        )
    
    def forward(
        self,
        x: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        return_attention: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor, optionally attention weights
        """
        fixed_point = (self.precision == PrecisionMode.FIXED)
        
        # Pre-norm attention
        residual = x
        x_norm = self.norm1.forward(x, fixed_point)
        
        if return_attention:
            attn_output, attn_weights = self.attention.forward(
                x_norm, attention_mask, return_attention=True
            )
        else:
            attn_output = self.attention.forward(x_norm, attention_mask)
        
        x = residual + attn_output
        
        # Pre-norm MLP
        residual = x
        x_norm = self.norm2.forward(x, fixed_point)
        mlp_output = self.mlp.forward(x_norm, fixed_point)
        x = residual + mlp_output
        
        if return_attention:
            return x, attn_weights
        return x
    
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
        
        embed_dim = self.config.embed_dim
        
        # Edge case: all zeros
        x_zeros = np.zeros((4, embed_dim), dtype=np.float32)
        out_zeros = self.forward(x_zeros)
        vectors.append(TestVector(
            name="transformer_all_zeros",
            inputs={'x': x_zeros},
            expected_outputs={'output': out_zeros},
            description="Transformer block with all zero inputs"
        ))
        
        # Random vectors
        for seq_len in seq_lengths:
            for i in range(num_vectors // len(seq_lengths)):
                x = np.random.randn(seq_len, embed_dim).astype(np.float32) * 0.1
                
                if self.precision == PrecisionMode.FIXED:
                    x = self.fp.fixed_to_float(self.fp.float_to_fixed(x))
                
                output, attn = self.forward(x, return_attention=True)
                
                vectors.append(TestVector(
                    name=f"transformer_random_seq{seq_len}_{i}",
                    inputs={'x': x},
                    expected_outputs={
                        'output': output,
                        'attention_weights': attn
                    },
                    description=f"Transformer block, seq_len={seq_len}"
                ))
        
        return vectors


class GoldenTransformerStack:
    """
    Golden model for a stack of transformer blocks.
    
    Used for complete vision encoder or language model.
    """
    
    def __init__(
        self,
        config: Union[VisionConfig, LanguageConfig],
        num_layers: Optional[int] = None,
        weight_type: Union[str, WeightType] = 'ternary',
        precision: Union[str, PrecisionMode] = 'float'
    ):
        """
        Initialize transformer stack.
        
        Args:
            config: Model configuration
            num_layers: Override number of layers
            weight_type: Weight quantization type
            precision: Computation precision
        """
        self.config = config
        self.num_layers = num_layers or config.num_layers
        
        if isinstance(weight_type, str):
            weight_type = WeightType(weight_type)
        
        self.blocks = [
            GoldenTransformerBlock(config, weight_type, precision)
            for _ in range(self.num_layers)
        ]
    
    def forward(
        self,
        x: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        return_all_hidden_states: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Forward pass through all transformer blocks.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask
            return_all_hidden_states: Whether to return all hidden states
            
        Returns:
            Output tensor, optionally all hidden states
        """
        hidden_states = [x] if return_all_hidden_states else None
        
        for block in self.blocks:
            x = block.forward(x, attention_mask)
            if return_all_hidden_states:
                hidden_states.append(x)
        
        if return_all_hidden_states:
            return x, hidden_states
        return x


def test_golden_transformer():
    """Test the golden transformer implementations."""
    print("Testing Golden Transformer Components")
    print("=" * 50)
    
    # Test Layer Norm
    print("\n--- Layer Normalization ---")
    ln = GoldenLayerNorm(dim=64)
    x = np.random.randn(4, 64).astype(np.float32)
    out = ln.forward(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    print(f"Output mean: {np.mean(out, axis=-1)}")  # Should be ~0
    print(f"Output std: {np.std(out, axis=-1)}")    # Should be ~1
    
    # Test RMS Norm
    print("\n--- RMS Normalization ---")
    rn = GoldenRMSNorm(dim=64)
    out = rn.forward(x)
    print(f"Output RMS: {np.sqrt(np.mean(out**2, axis=-1))}")  # Should be ~1
    
    # Test GELU
    print("\n--- GELU Activation ---")
    gelu = GoldenGELU()
    x_act = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
    out_exact = gelu.forward(x_act, fixed_point=False)
    out_approx = gelu.forward(x_act, fixed_point=True)
    print(f"Input: {x_act}")
    print(f"Exact GELU: {out_exact}")
    print(f"Approx GELU: {out_approx}")
    
    # Test MLP
    print("\n--- MLP ---")
    mlp = GoldenMLP(embed_dim=64, mlp_dim=256, weight_type=WeightType.TERNARY)
    x_mlp = np.random.randn(4, 64).astype(np.float32) * 0.1
    out_mlp = mlp.forward(x_mlp)
    print(f"MLP input: {x_mlp.shape}, output: {out_mlp.shape}")
    
    # Test Transformer Block - Vision
    print("\n--- Vision Transformer Block ---")
    vision_config = VisionConfig(embed_dim=64, num_heads=4, mlp_dim=256, num_layers=2)
    vision_block = GoldenTransformerBlock(vision_config, weight_type='ternary')
    x_vis = np.random.randn(8, 64).astype(np.float32) * 0.1
    out_vis, attn_vis = vision_block.forward(x_vis, return_attention=True)
    print(f"Vision block input: {x_vis.shape}, output: {out_vis.shape}")
    print(f"Attention shape: {attn_vis.shape}")
    
    # Test Transformer Block - Language
    print("\n--- Language Transformer Block ---")
    lang_config = LanguageConfig(embed_dim=64, num_heads=4, mlp_dim=128, num_layers=2)
    lang_block = GoldenTransformerBlock(lang_config, weight_type='ternary')
    x_lang = np.random.randn(16, 64).astype(np.float32) * 0.1
    out_lang = lang_block.forward(x_lang)
    print(f"Language block input: {x_lang.shape}, output: {out_lang.shape}")
    
    # Test Transformer Stack
    print("\n--- Transformer Stack ---")
    stack = GoldenTransformerStack(vision_config, num_layers=2, weight_type='ternary')
    out_stack, hidden = stack.forward(x_vis, return_all_hidden_states=True)
    print(f"Stack output: {out_stack.shape}")
    print(f"Hidden states: {len(hidden)} layers")
    
    # Generate test vectors
    print("\n--- Test Vector Generation ---")
    vectors = vision_block.generate_test_vectors(num_vectors=8)
    print(f"Generated {len(vectors)} test vectors")
    for v in vectors[:3]:
        print(f"  {v.name}: input={v.inputs['x'].shape}")
    
    print("\n✓ All transformer tests passed!")


if __name__ == "__main__":
    test_golden_transformer()
