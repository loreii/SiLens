"""
SiLens Golden Models
====================

Pure Python/NumPy reference implementations for RTL verification.

Modules:
- golden_attention: Multi-head attention with binary/ternary weights
- golden_transformer: Complete transformer block (LayerNorm, Attention, MLP)
- golden_inference: Full vision-language model inference pipeline
- golden_vision_encoder: Complete vision encoder with patch embedding
- golden_projector: Multimodal projector (vision-to-language mapping)
- golden_language_model: LLaMA-style language model decoder
- golden_normalization: RMSNorm and LayerNorm implementations

Usage:
    from tests.golden import GoldenMultiHeadAttention, GoldenTransformerBlock
    
    # Create golden model
    attn = GoldenMultiHeadAttention(embed_dim=768, num_heads=12)
    
    # Generate test vectors
    vectors = attn.generate_test_vectors(num_vectors=100)
"""

from .golden_attention import (
    GoldenMultiHeadAttention,
    GoldenSoftmax,
    WeightType,
    PrecisionMode,
    AttentionConfig,
    TestVector,
    FixedPointOps,
)

from .golden_transformer import (
    GoldenTransformerBlock,
    GoldenTransformerStack,
    GoldenLayerNorm,
    GoldenRMSNorm,
    GoldenGELU,
    GoldenSiLU,
    GoldenMLP,
    VisionConfig,
    LanguageConfig,
)

from .golden_inference import (
    GoldenVisionLanguageModel,
    GoldenPatchEmbedding,
    GoldenMultiModalProjector,
    InferenceConfig,
)

# New golden models
from .golden_vision_encoder import (
    GoldenVisionEncoder,
    GoldenViTBlock,
    VisionEncoderConfig,
)

from .golden_projector import (
    GoldenMultimodalProjector,
    GoldenLinearProjector,
    GoldenMLPProjector,
    ProjectorConfig,
)

from .golden_language_model import (
    GoldenLanguageModel,
    GoldenLLMBlock,
    GoldenSwiGLUMLP,
    GoldenRoPE,
    LanguageModelConfig,
)

from .golden_normalization import (
    GoldenRMSNorm as GoldenRMSNormV2,
    GoldenLayerNorm as GoldenLayerNormV2,
    NormConfig,
)

__all__ = [
    # Attention
    'GoldenMultiHeadAttention',
    'GoldenSoftmax',
    'WeightType',
    'PrecisionMode',
    'AttentionConfig',
    'TestVector',
    'FixedPointOps',
    # Transformer
    'GoldenTransformerBlock',
    'GoldenTransformerStack',
    'GoldenLayerNorm',
    'GoldenRMSNorm',
    'GoldenGELU',
    'GoldenSiLU',
    'GoldenMLP',
    'VisionConfig',
    'LanguageConfig',
    # Inference
    'GoldenVisionLanguageModel',
    'GoldenPatchEmbedding',
    'GoldenMultiModalProjector',
    'InferenceConfig',
    # Vision Encoder
    'GoldenVisionEncoder',
    'GoldenViTBlock',
    'VisionEncoderConfig',
    # Projector
    'GoldenMultimodalProjector',
    'GoldenLinearProjector',
    'GoldenMLPProjector',
    'ProjectorConfig',
    # Language Model
    'GoldenLanguageModel',
    'GoldenLLMBlock',
    'GoldenSwiGLUMLP',
    'GoldenRoPE',
    'LanguageModelConfig',
    # Normalization
    'GoldenRMSNormV2',
    'GoldenLayerNormV2',
    'NormConfig',
]
