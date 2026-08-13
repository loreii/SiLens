#!/usr/bin/env python3
"""
SiLens Integration Test - End-to-End Pipeline
==============================================

Tests the complete inference pipeline from image to text generation.

Usage:
    pytest tests/integration/test_end_to_end.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestEndToEndInference:
    """Test complete inference pipeline."""
    
    def test_golden_model_pipeline(self):
        """Test golden model end-to-end inference."""
        try:
            from tests.golden.golden_inference import (
                GoldenVisionLanguageModel,
                InferenceConfig
            )
            from tests.golden.golden_transformer import VisionConfig, LanguageConfig
            from tests.golden.golden_attention import WeightType
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
        
        # Small config for testing
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
        
        model = GoldenVisionLanguageModel(config)
        
        # Test image
        np.random.seed(42)
        image = np.random.rand(56, 56, 3).astype(np.float32)
        
        # Run inference
        output, layers = model.forward(image)
        
        # Verify output
        assert output is not None, "No output produced"
        assert not np.isnan(output).any(), "Output contains NaN"
        assert not np.isinf(output).any(), "Output contains Inf"
        
        # Verify intermediate layers
        assert 'patch_embeddings' in layers
        assert 'vision_features' in layers
        assert 'projected_features' in layers

    def test_vision_encoder_pipeline(self):
        """Test vision encoder in isolation."""
        try:
            from tests.golden.golden_vision_encoder import GoldenVisionEncoder
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
        
        encoder = GoldenVisionEncoder(
            image_size=56,
            patch_size=14,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            weight_type='ternary'
        )
        
        # Test different image types
        test_images = [
            np.zeros((56, 56, 3), dtype=np.float32),
            np.ones((56, 56, 3), dtype=np.float32),
            np.random.rand(56, 56, 3).astype(np.float32)
        ]
        
        for i, image in enumerate(test_images):
            features = encoder.forward(image)
            
            # Check output shape
            expected_seq_len = (56 // 14) ** 2 + 1  # patches + CLS
            assert features.shape[0] == expected_seq_len, \
                f"Image {i}: Expected seq_len={expected_seq_len}, got {features.shape[0]}"
            assert features.shape[1] == 64, \
                f"Image {i}: Expected embed_dim=64, got {features.shape[1]}"
    
    def test_projector_pipeline(self):
        """Test projector in isolation."""
        try:
            from tests.golden.golden_projector import GoldenMultimodalProjector
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
        
        projector = GoldenMultimodalProjector(
            vision_dim=64,
            language_dim=32,
            hidden_dim=128,
            projector_type='mlp',
            weight_type='ternary'
        )
        
        # Test with different sequence lengths
        for seq_len in [1, 17, 64]:
            x = np.random.randn(seq_len, 64).astype(np.float32) * 0.1
            output = projector.forward(x)
            
            assert output.shape == (seq_len, 32), \
                f"seq_len={seq_len}: Expected shape ({seq_len}, 32), got {output.shape}"
    
    def test_language_model_pipeline(self):
        """Test language model in isolation."""
        try:
            from tests.golden.golden_language_model import GoldenLanguageModel
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
        
        model = GoldenLanguageModel(
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            weight_type='ternary'
        )
        
        # Test with different sequence lengths
        for seq_len in [1, 8, 16]:
            x = np.random.randn(seq_len, 64).astype(np.float32) * 0.1
            logits = model.forward(inputs_embeds=x)
            
            assert logits.shape == (seq_len, 1000), \
                f"seq_len={seq_len}: Expected shape ({seq_len}, 1000), got {logits.shape}"
    
    def test_full_pipeline_deterministic(self):
        """Test that full pipeline produces deterministic results."""
        try:
            from tests.golden.golden_inference import (
                GoldenVisionLanguageModel,
                InferenceConfig
            )
            from tests.golden.golden_transformer import VisionConfig, LanguageConfig
            from tests.golden.golden_attention import WeightType
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
        
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
        
        # Same seed, same model
        np.random.seed(100)
        model1 = GoldenVisionLanguageModel(config)
        
        np.random.seed(100)
        model2 = GoldenVisionLanguageModel(config)
        
        # Same input
        image = np.random.rand(56, 56, 3).astype(np.float32)
        
        output1, _ = model1.forward(image)
        output2, _ = model2.forward(image)
        
        assert np.allclose(output1, output2), "Results not deterministic"


class TestPipelineComponents:
    """Test individual pipeline components."""
    
    def test_normalization_components(self):
        """Test normalization layers."""
        try:
            from tests.golden.golden_normalization import (
                GoldenRMSNorm, GoldenLayerNorm
            )
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
        
        dim = 64
        
        # Test RMSNorm
        rms_norm = GoldenRMSNorm(dim=dim)
        x = np.random.randn(16, dim).astype(np.float32)
        out = rms_norm.forward(x)
        
        # Output should have unit RMS (approximately)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=0.1), "RMSNorm output RMS not ~1"
        
        # Test LayerNorm
        layer_norm = GoldenLayerNorm(dim=dim)
        out = layer_norm.forward(x)
        
        # Output should have mean ~0, std ~1
        mean = np.mean(out, axis=-1)
        std = np.std(out, axis=-1)
        assert np.allclose(mean, 0.0, atol=0.01), "LayerNorm mean not ~0"
        assert np.allclose(std, 1.0, atol=0.1), "LayerNorm std not ~1"
    
    def test_attention_component(self):
        """Test attention mechanism."""
        try:
            from tests.golden.golden_attention import GoldenMultiHeadAttention
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
        
        attn = GoldenMultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            weight_type='ternary'
        )
        
        x = np.random.randn(8, 64).astype(np.float32) * 0.1
        output, weights = attn.forward(x, return_attention=True)
        
        # Output shape should match input
        assert output.shape == x.shape
        
        # Attention weights should sum to 1
        attn_sum = np.sum(weights, axis=-1)
        assert np.allclose(attn_sum, 1.0, atol=0.1), "Attention weights don't sum to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
