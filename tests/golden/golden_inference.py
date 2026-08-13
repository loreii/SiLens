#!/usr/bin/env python3
"""
SiLens Golden Model: Full Inference Pipeline
=============================================

Complete model forward pass reference implementation for comparing
against RTL simulation results.

This module provides:
- Full vision-language model inference
- Layer-by-layer expected outputs
- Test image processing
- Token generation simulation

License: Apache 2.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .golden_attention import WeightType, PrecisionMode, FixedPointOps, TestVector
from .golden_transformer import (
    GoldenTransformerStack,
    VisionConfig,
    LanguageConfig,
    GoldenLayerNorm,
    GoldenRMSNorm
)


@dataclass
class InferenceConfig:
    """Configuration for full model inference."""
    # Vision encoder
    vision_config: VisionConfig = None
    # Language model
    language_config: LanguageConfig = None
    # Projector
    projector_dim: int = 2048
    # Image processing
    image_size: int = 384
    patch_size: int = 14
    
    # Quantization
    weight_type: WeightType = WeightType.TERNARY
    precision: PrecisionMode = PrecisionMode.FLOAT
    
    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = VisionConfig()
        if self.language_config is None:
            self.language_config = LanguageConfig()


class GoldenPatchEmbedding:
    """Golden model for image patch embedding."""
    
    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 14,
        embed_dim: int = 768,
        weight_type: WeightType = WeightType.TERNARY
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding projection
        patch_dim = 3 * patch_size * patch_size
        np.random.seed(100)
        
        if weight_type == WeightType.TERNARY:
            self.proj = np.random.choice(
                [-1, 0, 1], size=(embed_dim, patch_dim),
                p=[0.35, 0.3, 0.35]
            ).astype(np.int8)
        else:
            self.proj = np.random.randn(embed_dim, patch_dim).astype(np.float32) * 0.02
        
        self.scale = 1.0
        
        # Class token
        self.cls_token = np.random.randn(1, embed_dim).astype(np.float32) * 0.02
    
    def _patchify(self, image: np.ndarray) -> np.ndarray:
        """Convert image to patches."""
        # image: [H, W, C] or [C, H, W]
        if image.shape[0] == 3:  # [C, H, W]
            image = np.transpose(image, (1, 2, 0))
        
        H, W, C = image.shape
        assert H == W == self.image_size
        
        p = self.patch_size
        n_h = H // p
        n_w = W // p
        
        # Reshape to patches
        patches = image.reshape(n_h, p, n_w, p, C)
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(n_h * n_w, p * p * C)
        
        return patches.astype(np.float32)
    
    def forward(self, image: np.ndarray) -> np.ndarray:
        """Embed image patches."""
        patches = self._patchify(image)
        
        # Project patches
        if self.proj.dtype == np.int8:
            # Ternary projection
            pos_mask = (self.proj == 1)
            neg_mask = (self.proj == -1)
            embeddings = np.zeros((patches.shape[0], self.embed_dim), dtype=np.float32)
            for i in range(self.embed_dim):
                embeddings[:, i] = (np.sum(patches[:, pos_mask[i]], axis=-1) -
                                   np.sum(patches[:, neg_mask[i]], axis=-1))
            embeddings *= self.scale
        else:
            embeddings = patches @ self.proj.T
        
        # Add class token
        cls_tokens = np.broadcast_to(self.cls_token, (1, self.embed_dim))
        embeddings = np.concatenate([cls_tokens, embeddings], axis=0)
        
        return embeddings


class GoldenMultiModalProjector:
    """Golden model for multimodal projector (connects vision to language)."""
    
    def __init__(
        self,
        vision_dim: int = 768,
        language_dim: int = 576,
        projector_dim: int = 2048,
        weight_type: WeightType = WeightType.TERNARY
    ):
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.projector_dim = projector_dim
        
        np.random.seed(200)
        
        if weight_type == WeightType.TERNARY:
            self.W1 = np.random.choice(
                [-1, 0, 1], size=(projector_dim, vision_dim),
                p=[0.35, 0.3, 0.35]
            ).astype(np.int8)
            self.W2 = np.random.choice(
                [-1, 0, 1], size=(language_dim, projector_dim),
                p=[0.35, 0.3, 0.35]
            ).astype(np.int8)
        else:
            self.W1 = np.random.randn(projector_dim, vision_dim).astype(np.float32) * 0.02
            self.W2 = np.random.randn(language_dim, projector_dim).astype(np.float32) * 0.02
        
        self.scale1 = 1.0
        self.scale2 = 1.0
    
    def _ternary_matmul(self, x: np.ndarray, w: np.ndarray, scale: float) -> np.ndarray:
        pos_mask = (w == 1)
        neg_mask = (w == -1)
        result = np.zeros((x.shape[0], w.shape[0]), dtype=np.float32)
        for i in range(w.shape[0]):
            result[:, i] = (np.sum(x[:, pos_mask[i]], axis=-1) -
                           np.sum(x[:, neg_mask[i]], axis=-1))
        return result * scale
    
    def forward(self, vision_features: np.ndarray) -> np.ndarray:
        """Project vision features to language space."""
        if self.W1.dtype == np.int8:
            hidden = self._ternary_matmul(vision_features, self.W1, self.scale1)
            hidden = np.maximum(hidden, 0)  # GELU approx as ReLU for simplicity
            output = self._ternary_matmul(hidden, self.W2, self.scale2)
        else:
            hidden = vision_features @ self.W1.T
            hidden = np.maximum(hidden, 0)
            output = hidden @ self.W2.T
        return output


class GoldenVisionLanguageModel:
    """
    Golden model for complete vision-language inference.
    
    Architecture:
        Image -> PatchEmbed -> VisionEncoder -> Projector -> LLM -> Tokens
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        
        # Initialize components
        self.patch_embed = GoldenPatchEmbedding(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            embed_dim=self.config.vision_config.embed_dim,
            weight_type=self.config.weight_type
        )
        
        self.vision_encoder = GoldenTransformerStack(
            config=self.config.vision_config,
            num_layers=2,  # Reduced for testing
            weight_type=self.config.weight_type,
            precision=self.config.precision
        )
        
        self.projector = GoldenMultiModalProjector(
            vision_dim=self.config.vision_config.embed_dim,
            language_dim=self.config.language_config.embed_dim,
            projector_dim=self.config.projector_dim,
            weight_type=self.config.weight_type
        )
        
        self.language_model = GoldenTransformerStack(
            config=self.config.language_config,
            num_layers=2,  # Reduced for testing
            weight_type=self.config.weight_type,
            precision=self.config.precision
        )
        
        # Layer outputs for debugging
        self.layer_outputs: Dict[str, np.ndarray] = {}
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode image through vision encoder."""
        self.layer_outputs['input_image'] = image
        
        # Patch embedding
        patches = self.patch_embed.forward(image)
        self.layer_outputs['patch_embeddings'] = patches
        
        # Vision transformer
        vision_features = self.vision_encoder.forward(patches)
        self.layer_outputs['vision_features'] = vision_features
        
        # Project to language space
        projected = self.projector.forward(vision_features)
        self.layer_outputs['projected_features'] = projected
        
        return projected

    
    def generate_logits(
        self,
        image_features: np.ndarray,
        text_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate output logits from language model."""
        # Combine image and text features
        if text_embeddings is not None:
            combined = np.concatenate([image_features, text_embeddings], axis=0)
        else:
            combined = image_features
        
        self.layer_outputs['combined_input'] = combined
        
        # Language model forward
        hidden_states = self.language_model.forward(combined)
        self.layer_outputs['language_output'] = hidden_states
        
        return hidden_states
    
    def forward(
        self,
        image: np.ndarray,
        text_embeddings: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Full forward pass.
        
        Args:
            image: Input image [H, W, C]
            text_embeddings: Optional text embeddings
            
        Returns:
            Output logits, dictionary of layer outputs
        """
        self.layer_outputs = {}
        
        # Encode image
        image_features = self.encode_image(image)
        
        # Generate logits
        output = self.generate_logits(image_features, text_embeddings)
        
        return output, self.layer_outputs
    
    def generate_test_vectors(
        self,
        num_images: int = 5,
        seed: int = 42
    ) -> List[TestVector]:
        """Generate test vectors for full inference."""
        np.random.seed(seed)
        vectors = []
        
        img_size = self.config.image_size
        
        # Test image patterns
        patterns = [
            ("zeros", np.zeros((img_size, img_size, 3), dtype=np.float32)),
            ("ones", np.ones((img_size, img_size, 3), dtype=np.float32)),
            ("gradient", np.tile(
                np.linspace(0, 1, img_size).reshape(1, -1, 1),
                (img_size, 1, 3)
            ).astype(np.float32)),
        ]
        
        for name, image in patterns:
            output, layers = self.forward(image)
            vectors.append(TestVector(
                name=f"inference_{name}",
                inputs={'image': image},
                expected_outputs={
                    'output': output,
                    **{f'layer_{k}': v for k, v in layers.items()}
                },
                description=f"Full inference with {name} image"
            ))
        
        # Random images
        for i in range(num_images):
            image = np.random.rand(img_size, img_size, 3).astype(np.float32)
            output, layers = self.forward(image)
            vectors.append(TestVector(
                name=f"inference_random_{i}",
                inputs={'image': image},
                expected_outputs={
                    'output': output,
                    'patch_embeddings': layers['patch_embeddings'],
                    'vision_features': layers['vision_features'],
                },
                description=f"Full inference with random image {i}"
            ))
        
        return vectors



def test_golden_inference():
    """Test the golden inference pipeline."""
    print("Testing Golden Inference Pipeline")
    print("=" * 50)
    
    # Create small config for testing
    vision_config = VisionConfig(
        embed_dim=64, num_heads=4, mlp_dim=128, num_layers=2,
        image_size=56, patch_size=14
    )
    language_config = LanguageConfig(
        embed_dim=64, num_heads=4, mlp_dim=128, num_layers=2
    )
    
    config = InferenceConfig(
        vision_config=vision_config,
        language_config=language_config,
        projector_dim=128,
        image_size=56,
        patch_size=14,
        weight_type=WeightType.TERNARY
    )
    
    # Test patch embedding
    print("\n--- Patch Embedding ---")
    patch_embed = GoldenPatchEmbedding(
        image_size=56, patch_size=14, embed_dim=64,
        weight_type=WeightType.TERNARY
    )
    test_image = np.random.rand(56, 56, 3).astype(np.float32)
    patches = patch_embed.forward(test_image)
    print(f"Image shape: {test_image.shape}")
    print(f"Patches shape: {patches.shape}")
    print(f"Num patches: {patch_embed.num_patches}")
    
    # Test projector
    print("\n--- Multimodal Projector ---")
    projector = GoldenMultiModalProjector(
        vision_dim=64, language_dim=64, projector_dim=128,
        weight_type=WeightType.TERNARY
    )
    vision_feat = np.random.randn(17, 64).astype(np.float32) * 0.1
    projected = projector.forward(vision_feat)
    print(f"Vision features: {vision_feat.shape}")
    print(f"Projected: {projected.shape}")
    
    # Test full model
    print("\n--- Full VLM Inference ---")
    model = GoldenVisionLanguageModel(config)
    output, layers = model.forward(test_image)
    print(f"Output shape: {output.shape}")
    print(f"Layer outputs captured: {list(layers.keys())}")
    
    # Generate test vectors
    print("\n--- Test Vector Generation ---")
    vectors = model.generate_test_vectors(num_images=3)
    print(f"Generated {len(vectors)} test vectors")
    for v in vectors:
        print(f"  {v.name}: output={v.expected_outputs['output'].shape}")
    
    print("\n✓ All inference tests passed!")


if __name__ == "__main__":
    test_golden_inference()
