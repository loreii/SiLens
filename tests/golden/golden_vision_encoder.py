#!/usr/bin/env python3
"""
SiLens Golden Model: Vision Encoder
====================================

Complete vision encoder golden model for RTL verification.

This module provides:
- Patch embedding golden model
- Vision Transformer (ViT) encoder
- SigLIP-style architecture
- Test vector generation

Usage:
    from golden_vision_encoder import GoldenVisionEncoder
    
    encoder = GoldenVisionEncoder(
        image_size=384,
        patch_size=14,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        weight_type='ternary'
    )
    
    features = encoder.forward(image)

License: Apache 2.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

from .golden_attention import (
    WeightType,
    PrecisionMode,
    FixedPointOps,
    TestVector,
    GoldenMultiHeadAttention
)
from .golden_transformer import (
    VisionConfig,
    GoldenLayerNorm,
    GoldenMLP,
    GoldenGELU
)


@dataclass
class VisionEncoderConfig:
    """Configuration for vision encoder."""
    image_size: int = 384
    patch_size: int = 14
    num_channels: int = 3
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: int = 4
    
    # Derived values
    num_patches: int = None
    mlp_dim: int = None
    head_dim: int = None
    
    # Normalization
    norm_eps: float = 1e-6
    
    # Fixed-point parameters
    act_width: int = 8
    frac_bits: int = 4
    
    def __post_init__(self):
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.mlp_dim = self.embed_dim * self.mlp_ratio
        self.head_dim = self.embed_dim // self.num_heads


class GoldenPatchEmbedding:
    """
    Golden model for patch embedding layer.
    
    Converts image into sequence of patch embeddings.
    """
    
    def __init__(
        self,
        config: VisionEncoderConfig,
        weight_type: WeightType = WeightType.TERNARY,
        fp_ops: Optional[FixedPointOps] = None
    ):
        self.config = config
        self.weight_type = weight_type
        self.fp = fp_ops
        
        # Projection weight: [embed_dim, channels * patch_size * patch_size]
        patch_dim = config.num_channels * config.patch_size * config.patch_size
        self._init_weights(patch_dim)
    
    def _init_weights(self, patch_dim: int):
        """Initialize projection weights."""
        np.random.seed(100)
        
        if self.weight_type == WeightType.TERNARY:
            self.proj_weight = np.random.choice(
                [-1, 0, 1],
                size=(self.config.embed_dim, patch_dim),
                p=[0.35, 0.3, 0.35]
            ).astype(np.int8)
            self.scale = 0.02
        elif self.weight_type == WeightType.BINARY:
            self.proj_weight = np.random.choice(
                [-1, 1],
                size=(self.config.embed_dim, patch_dim)
            ).astype(np.int8)
            self.scale = 0.02
        else:
            self.proj_weight = np.random.randn(
                self.config.embed_dim, patch_dim
            ).astype(np.float32) * 0.02
            self.scale = 1.0
        
        # Position embeddings: [1 + num_patches, embed_dim]
        num_positions = 1 + self.config.num_patches  # +1 for CLS token
        self.position_embedding = np.random.randn(
            num_positions, self.config.embed_dim
        ).astype(np.float32) * 0.02
        
        # Class token: [1, embed_dim]
        self.cls_token = np.random.randn(
            1, self.config.embed_dim
        ).astype(np.float32) * 0.02
    
    def _patchify(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to patches.
        
        Args:
            image: [H, W, C] or [C, H, W]
            
        Returns:
            patches: [num_patches, patch_dim]
        """
        # Ensure image is [H, W, C]
        if image.shape[0] == self.config.num_channels:
            image = np.transpose(image, (1, 2, 0))
        
        H, W, C = image.shape
        p = self.config.patch_size
        n_h = H // p
        n_w = W // p
        
        # Extract patches
        patches = image.reshape(n_h, p, n_w, p, C)
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(n_h * n_w, p * p * C)
        
        return patches.astype(np.float32)
    
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
    
    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Forward pass through patch embedding.
        
        Args:
            image: Input image [H, W, C] or [C, H, W]
            
        Returns:
            embeddings: [1 + num_patches, embed_dim]
        """
        # Convert to patches
        patches = self._patchify(image)
        
        # Project patches
        if self.weight_type in [WeightType.TERNARY, WeightType.BINARY]:
            patch_embeddings = self._ternary_matmul(patches, self.proj_weight)
        else:
            patch_embeddings = patches @ self.proj_weight.T
        
        # Prepend CLS token
        batch_cls = np.broadcast_to(self.cls_token, (1, self.config.embed_dim))
        embeddings = np.concatenate([batch_cls, patch_embeddings], axis=0)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embedding
        
        return embeddings
    
    def generate_test_vectors(
        self,
        num_vectors: int = 10,
        seed: int = 42
    ) -> List[TestVector]:
        """Generate test vectors for RTL verification."""
        np.random.seed(seed)
        vectors = []
        
        img_size = self.config.image_size
        
        # All zeros image
        image = np.zeros((img_size, img_size, 3), dtype=np.float32)
        output = self.forward(image)
        vectors.append(TestVector(
            name="patch_embed_zeros",
            inputs={'image': image},
            expected_outputs={'embeddings': output},
            description="All zero image"
        ))
        
        # All ones image
        image = np.ones((img_size, img_size, 3), dtype=np.float32)
        output = self.forward(image)
        vectors.append(TestVector(
            name="patch_embed_ones",
            inputs={'image': image},
            expected_outputs={'embeddings': output},
            description="All ones image"
        ))
        
        # Gradient image
        gradient = np.tile(
            np.linspace(0, 1, img_size).reshape(1, -1, 1),
            (img_size, 1, 3)
        ).astype(np.float32)
        output = self.forward(gradient)
        vectors.append(TestVector(
            name="patch_embed_gradient",
            inputs={'image': gradient},
            expected_outputs={'embeddings': output},
            description="Horizontal gradient image"
        ))
        
        # Random images
        for i in range(num_vectors - 3):
            image = np.random.rand(img_size, img_size, 3).astype(np.float32)
            output = self.forward(image)
            vectors.append(TestVector(
                name=f"patch_embed_random_{i}",
                inputs={'image': image},
                expected_outputs={'embeddings': output},
                description=f"Random image {i}"
            ))
        
        return vectors


class GoldenViTBlock:
    """
    Golden model for a single Vision Transformer block.
    
    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """
    
    def __init__(
        self,
        config: VisionEncoderConfig,
        weight_type: WeightType = WeightType.TERNARY,
        precision: PrecisionMode = PrecisionMode.FLOAT
    ):
        self.config = config
        self.weight_type = weight_type
        self.precision = precision
        
        self.fp = FixedPointOps(width=config.act_width, frac_bits=config.frac_bits)
        
        # Layer normalization
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
            activation="gelu",
            weight_type=weight_type,
            use_swiglu=False,
            fp_ops=self.fp
        )
    
    def forward(
        self,
        x: np.ndarray,
        return_attention: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass through ViT block.
        
        Args:
            x: Input [seq_len, embed_dim]
            return_attention: Return attention weights
            
        Returns:
            Output, optionally attention weights
        """
        fixed_point = (self.precision == PrecisionMode.FIXED)
        
        # Pre-norm attention
        residual = x
        x_norm = self.norm1.forward(x, fixed_point)
        
        if return_attention:
            attn_out, attn_weights = self.attention.forward(
                x_norm, return_attention=True
            )
        else:
            attn_out = self.attention.forward(x_norm)
        
        x = residual + attn_out
        
        # Pre-norm MLP
        residual = x
        x_norm = self.norm2.forward(x, fixed_point)
        mlp_out = self.mlp.forward(x_norm, fixed_point)
        x = residual + mlp_out
        
        if return_attention:
            return x, attn_weights
        return x


class GoldenVisionEncoder:
    """
    Complete vision encoder golden model.
    
    Architecture:
        - Patch embedding (image -> patches -> embeddings)
        - CLS token + position embeddings
        - N x ViT blocks
        - Layer normalization
        - Output: CLS token or all tokens
    """
    
    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 14,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        weight_type: Union[str, WeightType] = 'ternary',
        precision: Union[str, PrecisionMode] = 'float'
    ):
        if isinstance(weight_type, str):
            weight_type = WeightType(weight_type)
        if isinstance(precision, str):
            precision = PrecisionMode(precision)
        
        self.config = VisionEncoderConfig(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        self.weight_type = weight_type
        self.precision = precision
        
        self.fp = FixedPointOps(
            width=self.config.act_width,
            frac_bits=self.config.frac_bits
        )
        
        # Patch embedding
        self.patch_embed = GoldenPatchEmbedding(
            self.config, weight_type, self.fp
        )
        
        # Transformer blocks
        self.blocks = [
            GoldenViTBlock(self.config, weight_type, precision)
            for _ in range(num_layers)
        ]
        
        # Final layer normalization
        self.norm = GoldenLayerNorm(embed_dim, self.config.norm_eps, self.fp)
        
        # Store intermediate outputs for debugging
        self.layer_outputs: Dict[str, np.ndarray] = {}
    
    def forward(
        self,
        image: np.ndarray,
        return_all_tokens: bool = True,
        return_hidden_states: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Forward pass through vision encoder.
        
        Args:
            image: Input image [H, W, C] or [C, H, W]
            return_all_tokens: Return all tokens (True) or just CLS (False)
            return_hidden_states: Return all hidden states
            
        Returns:
            Features, optionally all hidden states
        """
        self.layer_outputs = {}
        fixed_point = (self.precision == PrecisionMode.FIXED)
        
        # Patch embedding
        x = self.patch_embed.forward(image)
        self.layer_outputs['patch_embeddings'] = x.copy()
        
        # Transformer blocks
        hidden_states = [x] if return_hidden_states else None
        
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
            self.layer_outputs[f'block_{i}_output'] = x.copy()
            
            if return_hidden_states:
                hidden_states.append(x.copy())
        
        # Final layer norm
        x = self.norm.forward(x, fixed_point)
        self.layer_outputs['final_output'] = x.copy()
        
        # Return CLS token or all tokens
        if not return_all_tokens:
            x = x[0:1]
        
        if return_hidden_states:
            return x, hidden_states
        return x
    
    def generate_test_vectors(
        self,
        num_images: int = 5,
        seed: int = 42
    ) -> List[TestVector]:
        """Generate test vectors for RTL verification."""
        np.random.seed(seed)
        vectors = []
        
        img_size = self.config.image_size
        
        # Standard test patterns
        patterns = [
            ("zeros", np.zeros((img_size, img_size, 3), dtype=np.float32)),
            ("ones", np.ones((img_size, img_size, 3), dtype=np.float32)),
            ("checkerboard", self._create_checkerboard(img_size)),
        ]
        
        for name, image in patterns:
            output = self.forward(image)
            vectors.append(TestVector(
                name=f"vision_encoder_{name}",
                inputs={'image': image},
                expected_outputs={
                    'output': output,
                    'patch_embeddings': self.layer_outputs['patch_embeddings']
                },
                description=f"Vision encoder with {name} image"
            ))
        
        # Random images
        for i in range(num_images):
            image = np.random.rand(img_size, img_size, 3).astype(np.float32)
            output = self.forward(image)
            vectors.append(TestVector(
                name=f"vision_encoder_random_{i}",
                inputs={'image': image},
                expected_outputs={
                    'output': output,
                    'patch_embeddings': self.layer_outputs['patch_embeddings']
                },
                description=f"Random image {i}"
            ))
        
        return vectors
    
    def _create_checkerboard(self, size: int, block_size: int = 14) -> np.ndarray:
        """Create a checkerboard pattern."""
        pattern = np.zeros((size, size, 3), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    pattern[i, j] = 1.0
        return pattern


def test_golden_vision_encoder():
    """Test the golden vision encoder implementation."""
    print("Testing Golden Vision Encoder")
    print("=" * 50)
    
    # Use small config for testing
    config = VisionEncoderConfig(
        image_size=56,
        patch_size=14,
        embed_dim=64,
        num_heads=4,
        num_layers=2
    )
    
    # Test patch embedding
    print("\n--- Patch Embedding ---")
    patch_embed = GoldenPatchEmbedding(config, WeightType.TERNARY)
    test_image = np.random.rand(56, 56, 3).astype(np.float32)
    embeddings = patch_embed.forward(test_image)
    
    print(f"Image shape: {test_image.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Expected patches: {config.num_patches + 1}")
    
    assert embeddings.shape[0] == config.num_patches + 1
    assert embeddings.shape[1] == config.embed_dim
    
    # Test ViT block
    print("\n--- ViT Block ---")
    block = GoldenViTBlock(config, WeightType.TERNARY)
    x = np.random.randn(17, 64).astype(np.float32) * 0.1
    output, attn = block.forward(x, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")
    
    assert output.shape == x.shape
    
    # Test full encoder
    print("\n--- Full Vision Encoder ---")
    encoder = GoldenVisionEncoder(
        image_size=56,
        patch_size=14,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        weight_type='ternary'
    )
    
    features, hidden = encoder.forward(test_image, return_hidden_states=True)
    
    print(f"Output shape: {features.shape}")
    print(f"Hidden states: {len(hidden)} layers")
    
    # Test vector generation
    print("\n--- Test Vector Generation ---")
    vectors = encoder.generate_test_vectors(num_images=3)
    print(f"Generated {len(vectors)} test vectors")
    
    print("\n✓ All vision encoder tests passed!")


if __name__ == "__main__":
    test_golden_vision_encoder()
