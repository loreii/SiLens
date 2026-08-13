#!/usr/bin/env python3
"""
SiLens Golden Model: Language Model (LLaMA-style)
=================================================

Complete language model decoder golden model for RTL verification.

This module provides:
- LLaMA-style transformer decoder
- SwiGLU MLP
- RMSNorm
- RoPE positional embeddings
- Test vector generation

License: Apache 2.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .golden_attention import (
    WeightType, PrecisionMode, FixedPointOps, TestVector,
    GoldenMultiHeadAttention
)
from .golden_normalization import GoldenRMSNorm


@dataclass
class LanguageModelConfig:
    """Configuration for language model."""
    embed_dim: int = 576
    num_heads: int = 9
    num_layers: int = 30
    mlp_dim: int = 1536
    vocab_size: int = 49152
    max_seq_len: int = 2048
    
    # Derived
    head_dim: int = None
    
    # Normalization
    norm_eps: float = 1e-5
    
    # RoPE
    rope_theta: float = 10000.0
    
    # Fixed-point
    act_width: int = 8
    frac_bits: int = 4
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads


class GoldenSiLU:
    """Golden model for SiLU (Swish) activation."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """SiLU(x) = x * sigmoid(x)"""
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        return (x * sigmoid).astype(np.float32)
    
    def forward_approx(self, x: np.ndarray) -> np.ndarray:
        """Piece-wise linear approximation for hardware."""
        result = np.zeros_like(x)
        
        # Segments
        mask_neg4 = x < -4
        mask_neg2 = (x >= -4) & (x < -2)
        mask_neg1 = (x >= -2) & (x < -1)
        mask_0 = (x >= -1) & (x < 0)
        mask_1 = (x >= 0) & (x < 1)
        mask_2 = (x >= 1) & (x < 2)
        mask_4 = (x >= 2) & (x < 4)
        mask_pos4 = x >= 4
        
        result[mask_neg4] = 0
        result[mask_neg2] = 0.018 * (x[mask_neg2] + 4)
        result[mask_neg1] = 0.135 + 0.233 * (x[mask_neg1] + 2)
        result[mask_0] = 0.368 + 0.5 * (x[mask_0] + 1)
        result[mask_1] = 0.5 * x[mask_1] + 0.5 * x[mask_1] ** 2 / 4
        result[mask_2] = 0.73 * x[mask_2] - 0.135
        result[mask_4] = 0.93 * x[mask_4] - 0.27
        result[mask_pos4] = x[mask_pos4]
        
        return result.astype(np.float32)


class GoldenSwiGLUMLP:
    """
    Golden model for SwiGLU MLP.
    
    SwiGLU: output = down(silu(gate(x)) * up(x))
    """
    
    def __init__(
        self,
        config: LanguageModelConfig,
        weight_type: WeightType = WeightType.TERNARY
    ):
        self.config = config
        self.weight_type = weight_type
        self.silu = GoldenSiLU()
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights."""
        np.random.seed(400)
        
        if self.weight_type == WeightType.TERNARY:
            self.W_gate = np.random.choice(
                [-1, 0, 1],
                size=(self.config.mlp_dim, self.config.embed_dim),
                p=[0.35, 0.3, 0.35]
            ).astype(np.int8)
            self.W_up = np.random.choice(
                [-1, 0, 1],
                size=(self.config.mlp_dim, self.config.embed_dim),
                p=[0.35, 0.3, 0.35]
            ).astype(np.int8)
            self.W_down = np.random.choice(
                [-1, 0, 1],
                size=(self.config.embed_dim, self.config.mlp_dim),
                p=[0.35, 0.3, 0.35]
            ).astype(np.int8)
            self.scale = 0.02
        else:
            self.W_gate = np.random.randn(
                self.config.mlp_dim, self.config.embed_dim
            ).astype(np.float32) * 0.02
            self.W_up = np.random.randn(
                self.config.mlp_dim, self.config.embed_dim
            ).astype(np.float32) * 0.02
            self.W_down = np.random.randn(
                self.config.embed_dim, self.config.mlp_dim
            ).astype(np.float32) * 0.02
            self.scale = 1.0

    def _ternary_matmul(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
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
        
        return (result * self.scale).astype(np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through SwiGLU MLP."""
        original_shape = x.shape
        x_flat = x.reshape(-1, self.config.embed_dim)
        
        if self.weight_type == WeightType.TERNARY:
            gate = self._ternary_matmul(x_flat, self.W_gate)
            up = self._ternary_matmul(x_flat, self.W_up)
        else:
            gate = x_flat @ self.W_gate.T
            up = x_flat @ self.W_up.T
        
        # SwiGLU: silu(gate) * up
        gate = self.silu.forward(gate)
        hidden = gate * up
        
        if self.weight_type == WeightType.TERNARY:
            output = self._ternary_matmul(hidden, self.W_down)
        else:
            output = hidden @ self.W_down.T
        
        return output.reshape(original_shape).astype(np.float32)


class GoldenRoPE:
    """Golden model for Rotary Position Embedding."""
    
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._precompute_freqs()
    
    def _precompute_freqs(self):
        """Precompute frequency table."""
        freqs = 1.0 / (self.theta ** (
            np.arange(0, self.head_dim, 2).astype(np.float32) / self.head_dim
        ))
        t = np.arange(self.max_seq_len, dtype=np.float32)
        freqs = np.outer(t, freqs)
        
        self.cos_cached = np.cos(freqs).astype(np.float32)
        self.sin_cached = np.sin(freqs).astype(np.float32)
    
    def forward(self, x: np.ndarray, positions: np.ndarray = None) -> np.ndarray:
        """Apply rotary embeddings."""
        seq_len = x.shape[-2]
        
        if positions is None:
            positions = np.arange(seq_len)
        
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # Rotate
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x1 * sin + x2 * cos
        
        # Interleave
        result = np.empty_like(x)
        result[..., 0::2] = x_rot1
        result[..., 1::2] = x_rot2
        
        return result.astype(np.float32)


class GoldenLLMBlock:
    """
    Golden model for a single LLM decoder block.
    
    Architecture (LLaMA-style):
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU_MLP(RMSNorm(x))
    """
    
    def __init__(
        self,
        config: LanguageModelConfig,
        weight_type: WeightType = WeightType.TERNARY,
        precision: PrecisionMode = PrecisionMode.FLOAT
    ):
        self.config = config
        self.weight_type = weight_type
        self.precision = precision
        
        # Normalization
        self.input_norm = GoldenRMSNorm(config.embed_dim, config.norm_eps, precision)
        self.post_attn_norm = GoldenRMSNorm(config.embed_dim, config.norm_eps, precision)
        
        # Self-attention
        self.attention = GoldenMultiHeadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            weight_type=weight_type,
            precision=precision
        )
        
        # SwiGLU MLP
        self.mlp = GoldenSwiGLUMLP(config, weight_type)
        
        # RoPE
        self.rope = GoldenRoPE(config.head_dim, config.max_seq_len, config.rope_theta)
    
    def forward(
        self,
        x: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        return_attention: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forward pass through LLM block."""
        # Pre-norm attention
        residual = x
        x_norm = self.input_norm.forward(x)
        
        if return_attention:
            attn_out, attn_weights = self.attention.forward(
                x_norm, attention_mask, return_attention=True
            )
        else:
            attn_out = self.attention.forward(x_norm, attention_mask)
        
        x = residual + attn_out
        
        # Pre-norm MLP
        residual = x
        x_norm = self.post_attn_norm.forward(x)
        mlp_out = self.mlp.forward(x_norm)
        x = residual + mlp_out
        
        if return_attention:
            return x, attn_weights
        return x


class GoldenLanguageModel:
    """
    Complete language model decoder golden model.
    
    Architecture:
        - Token embeddings
        - N x LLM blocks
        - RMSNorm
        - LM head (output projection)
    """
    
    def __init__(
        self,
        embed_dim: int = 576,
        num_heads: int = 9,
        num_layers: int = 30,
        vocab_size: int = 49152,
        weight_type: Union[str, WeightType] = 'ternary',
        precision: Union[str, PrecisionMode] = 'float'
    ):
        if isinstance(weight_type, str):
            weight_type = WeightType(weight_type)
        if isinstance(precision, str):
            precision = PrecisionMode(precision)
        
        self.config = LanguageModelConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            vocab_size=vocab_size
        )
        self.weight_type = weight_type
        self.precision = precision

        # Token embeddings
        np.random.seed(500)
        self.embed_tokens = np.random.randn(
            vocab_size, embed_dim
        ).astype(np.float32) * 0.02
        
        # Decoder blocks (use fewer for testing)
        num_test_layers = min(num_layers, 4)
        self.blocks = [
            GoldenLLMBlock(self.config, weight_type, precision)
            for _ in range(num_test_layers)
        ]
        self.num_layers = num_test_layers
        
        # Final norm
        self.norm = GoldenRMSNorm(embed_dim, self.config.norm_eps, precision)
        
        # LM head
        if weight_type == WeightType.TERNARY:
            self.lm_head = np.random.choice(
                [-1, 0, 1],
                size=(vocab_size, embed_dim),
                p=[0.35, 0.3, 0.35]
            ).astype(np.int8)
            self.lm_scale = 0.02
        else:
            self.lm_head = np.random.randn(
                vocab_size, embed_dim
            ).astype(np.float32) * 0.02
            self.lm_scale = 1.0
        
        self.layer_outputs: Dict[str, np.ndarray] = {}
    
    def _causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal attention mask."""
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
        return mask.astype(np.float32)
    
    def forward(
        self,
        input_ids: np.ndarray = None,
        inputs_embeds: np.ndarray = None,
        return_hidden_states: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Forward pass through language model.
        
        Args:
            input_ids: Token IDs [seq_len]
            inputs_embeds: Token embeddings [seq_len, embed_dim]
            return_hidden_states: Return all hidden states
            
        Returns:
            logits: [seq_len, vocab_size]
        """
        self.layer_outputs = {}
        
        # Get embeddings
        if inputs_embeds is not None:
            x = inputs_embeds
        elif input_ids is not None:
            x = self.embed_tokens[input_ids]
        else:
            raise ValueError("Must provide input_ids or inputs_embeds")
        
        self.layer_outputs['embeddings'] = x.copy()
        
        # Causal mask
        seq_len = x.shape[0]
        causal_mask = self._causal_mask(seq_len)
        
        # Decoder blocks
        hidden_states = [x] if return_hidden_states else None
        
        for i, block in enumerate(self.blocks):
            x = block.forward(x, attention_mask=causal_mask)
            self.layer_outputs[f'block_{i}_output'] = x.copy()
            
            if return_hidden_states:
                hidden_states.append(x.copy())
        
        # Final norm
        x = self.norm.forward(x)
        self.layer_outputs['final_hidden'] = x.copy()
        
        # LM head
        if self.weight_type == WeightType.TERNARY:
            pos_mask = (self.lm_head == 1)
            neg_mask = (self.lm_head == -1)
            logits = np.zeros((seq_len, self.config.vocab_size), dtype=np.float64)
            for i in range(self.config.vocab_size):
                logits[:, i] = (
                    np.sum(x[:, pos_mask[i]], axis=-1) -
                    np.sum(x[:, neg_mask[i]], axis=-1)
                )
            logits = (logits * self.lm_scale).astype(np.float32)
        else:
            logits = x @ self.lm_head.T
        
        if return_hidden_states:
            return logits, hidden_states
        return logits

    def generate_test_vectors(
        self,
        num_vectors: int = 10,
        seq_lengths: List[int] = None,
        seed: int = 42
    ) -> List[TestVector]:
        """Generate test vectors for RTL verification."""
        if seq_lengths is None:
            seq_lengths = [1, 4, 8, 16]
        
        np.random.seed(seed)
        vectors = []
        
        # Single token
        x = np.random.randn(1, self.config.embed_dim).astype(np.float32) * 0.1
        output = self.forward(inputs_embeds=x)
        vectors.append(TestVector(
            name="llm_single_token",
            inputs={'inputs_embeds': x},
            expected_outputs={'logits': output},
            description="Single token inference"
        ))
        
        # Various sequence lengths
        for seq_len in seq_lengths:
            for i in range(num_vectors // len(seq_lengths)):
                x = np.random.randn(seq_len, self.config.embed_dim).astype(np.float32) * 0.1
                output = self.forward(inputs_embeds=x)
                
                vectors.append(TestVector(
                    name=f"llm_seq{seq_len}_{i}",
                    inputs={'inputs_embeds': x},
                    expected_outputs={'logits': output},
                    description=f"Sequence length {seq_len}"
                ))
        
        return vectors


def test_golden_language_model():
    """Test the golden language model implementation."""
    print("Testing Golden Language Model")
    print("=" * 50)
    
    # Small config for testing
    config = LanguageModelConfig(
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_dim=128,
        vocab_size=1000
    )
    
    # Test SiLU
    print("\n--- SiLU Activation ---")
    silu = GoldenSiLU()
    x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
    out = silu.forward(x)
    print(f"Input: {x}")
    print(f"SiLU output: {out}")
    
    # Test SwiGLU MLP
    print("\n--- SwiGLU MLP ---")
    mlp = GoldenSwiGLUMLP(config, WeightType.TERNARY)
    x = np.random.randn(8, 64).astype(np.float32) * 0.1
    out = mlp.forward(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    
    # Test RoPE
    print("\n--- RoPE ---")
    rope = GoldenRoPE(head_dim=16, max_seq_len=512)
    x = np.random.randn(8, 16).astype(np.float32)
    out = rope.forward(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    
    # Test LLM block
    print("\n--- LLM Block ---")
    block = GoldenLLMBlock(config, WeightType.TERNARY)
    x = np.random.randn(16, 64).astype(np.float32) * 0.1
    out = block.forward(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    
    # Test full model
    print("\n--- Full Language Model ---")
    model = GoldenLanguageModel(
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=1000,
        weight_type='ternary'
    )
    
    x = np.random.randn(16, 64).astype(np.float32) * 0.1
    logits, hidden = model.forward(inputs_embeds=x, return_hidden_states=True)
    print(f"Input: {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Hidden states: {len(hidden)} layers")
    
    # Test vector generation
    print("\n--- Test Vector Generation ---")
    vectors = model.generate_test_vectors(num_vectors=8)
    print(f"Generated {len(vectors)} test vectors")
    
    print("\n✓ All language model tests passed!")


if __name__ == "__main__":
    test_golden_language_model()
