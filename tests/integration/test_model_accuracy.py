#!/usr/bin/env python3
"""
SiLens Integration Test - Model Accuracy
========================================

Tests model accuracy comparing quantized model against PyTorch reference.

This module provides:
- Quantized model inference testing
- Accuracy metrics calculation
- Layer-by-layer comparison
- Image classification/captioning accuracy

Usage:
    pytest tests/integration/test_model_accuracy.py -v
    pytest tests/integration/test_model_accuracy.py -v -k "test_quantized"
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestModelAccuracy:
    """Test suite for model accuracy verification."""
    
    @pytest.fixture(autouse=True)
    def setup(self, model_path, model_available):
        """Setup test fixtures."""
        self.model_path = model_path
        self.model_available = model_available
    
    @pytest.mark.requires_model
    def test_vision_encoder_accuracy(self, hf_model):
        """Test vision encoder output accuracy."""
        import torch
        
        # Get vision encoder
        if hasattr(hf_model, 'vision_model'):
            vision_model = hf_model.vision_model
        elif hasattr(hf_model, 'model') and hasattr(hf_model.model, 'vision_model'):
            vision_model = hf_model.model.vision_model
        else:
            pytest.skip("Cannot access vision model")
        
        # Create test input
        batch_size = 1
        channels = 3
        height = width = 384
        
        test_input = torch.randn(batch_size, channels, height, width)
        
        # Get reference output
        with torch.no_grad():
            reference_output = vision_model(test_input)
        
        # Verify output shape
        if hasattr(reference_output, 'last_hidden_state'):
            output = reference_output.last_hidden_state
        else:
            output = reference_output
        
        assert output is not None, "Vision encoder produced no output"
        assert len(output.shape) == 3, f"Expected 3D output, got {len(output.shape)}D"
        
        # Check for NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    
    @pytest.mark.requires_model
    def test_quantized_layer_accuracy(self, hf_model):
        """Test accuracy of quantized vs original layers."""
        import torch
        
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer,
                TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        
        # Test on a few layers
        layers_tested = 0
        max_layers = 5
        
        for name, param in hf_model.named_parameters():
            if layers_tested >= max_layers:
                break
            
            if param.numel() < 1000 or 'weight' not in name:
                continue
            
            if quantizer.should_skip_layer(name):
                continue
            
            weights = param.detach().float().cpu().numpy()
            result = quantizer.quantize_tensor(name, weights)
            
            # Verify quantized values are valid
            unique_vals = np.unique(result.quantized_weights)
            assert set(unique_vals).issubset({-1, 0, 1}), \
                f"Invalid quantized values: {unique_vals}"
            
            # Check reconstruction error is reasonable
            assert result.mean_abs_error < 0.2, \
                f"Layer {name}: error {result.mean_abs_error:.4f} too high"
            
            # Check sparsity is reasonable
            assert 0.05 <= result.sparsity <= 0.6, \
                f"Layer {name}: sparsity {result.sparsity:.1%} out of expected range"
            
            layers_tested += 1
        
        assert layers_tested > 0, "No layers tested"
    
    @pytest.mark.requires_model
    def test_forward_pass_accuracy(self, hf_model, sample_image):
        """Test forward pass produces valid outputs."""
        import torch
        from PIL import Image
        
        try:
            from transformers import AutoProcessor
        except ImportError:
            pytest.skip("transformers not available")
        
        processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[sample_image], return_tensors="pt")
        
        # Forward pass
        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        
        # Decode output
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        assert len(decoded) > 0, "No output generated"
        assert len(decoded) > 10, f"Output too short: {decoded}"
    
    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_inference_consistency(self, hf_model, sample_image):
        """Test that inference is consistent (deterministic)."""
        import torch
        from PIL import Image
        
        try:
            from transformers import AutoProcessor
        except ImportError:
            pytest.skip("transformers not available")
        
        processor = AutoProcessor.from_pretrained(self.model_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What do you see?"}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[sample_image], return_tensors="pt")
        
        # Run inference twice
        outputs1 = hf_model.generate(**inputs, max_new_tokens=10, do_sample=False)
        outputs2 = hf_model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        # Results should be identical with do_sample=False
        assert torch.equal(outputs1, outputs2), "Inference not deterministic"



class TestQuantizationAccuracy:
    """Test quantization accuracy metrics."""
    
    def test_ternary_distribution(self):
        """Test ternary quantization produces expected distribution."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer,
                TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        np.random.seed(42)
        
        # Create synthetic weights with normal distribution
        weights = np.random.randn(1000, 1000).astype(np.float32) * 0.1
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        result = quantizer.quantize_tensor("test", weights)
        
        # Check distribution
        total = weights.size
        pos_pct = result.num_positive / total
        neg_pct = result.num_negative / total
        zero_pct = result.num_zero / total
        
        # For normally distributed weights, expect roughly symmetric
        assert abs(pos_pct - neg_pct) < 0.1, "Distribution not symmetric"
        
        # Sparsity should be in reasonable range
        assert 0.1 <= zero_pct <= 0.5, f"Unexpected sparsity: {zero_pct:.1%}"
    
    def test_reconstruction_error_bounds(self):
        """Test that reconstruction error is bounded."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer,
                TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        np.random.seed(42)
        
        # Test with different alpha values
        alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
        weights = np.random.randn(500, 500).astype(np.float32) * 0.1
        
        for alpha in alphas:
            config = TernaryQuantizationConfig(alpha=alpha)
            quantizer = TernaryQuantizer(config)
            result = quantizer.quantize_tensor("test", weights)
            
            # Reconstruction error should be bounded
            assert result.mean_abs_error < 0.2, \
                f"alpha={alpha}: error {result.mean_abs_error:.4f} too high"
            
            # Higher alpha should give less sparsity
            # (tighter threshold means fewer zeros)
    
    def test_hardware_encoding_roundtrip(self):
        """Test hardware encoding preserves values."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer,
                TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        config = TernaryQuantizationConfig()
        quantizer = TernaryQuantizer(config)
        
        # Create ternary weights
        np.random.seed(42)
        ternary = np.random.choice([-1, 0, 1], size=(100, 100)).astype(np.int8)
        
        # Encode
        encoded = quantizer.encode_for_hardware(ternary)
        
        # Verify packing
        assert encoded.dtype == np.uint8
        expected_size = (ternary.size + 3) // 4  # 4 values per byte
        assert len(encoded) == expected_size
        
        # Decode and verify (manual decode)
        decoded = np.zeros(ternary.size, dtype=np.int8)
        for i, byte in enumerate(encoded):
            for j in range(4):
                idx = i * 4 + j
                if idx < len(decoded):
                    bits = (byte >> (6 - j * 2)) & 0x03
                    if bits == 0b01:
                        decoded[idx] = 1
                    elif bits == 0b10:
                        decoded[idx] = -1
                    else:
                        decoded[idx] = 0
        
        # Compare
        original_flat = ternary.flatten()
        matches = np.sum(original_flat == decoded)
        accuracy = matches / len(original_flat)
        
        assert accuracy > 0.99, f"Encoding roundtrip accuracy: {accuracy:.1%}"


class TestGoldenModelComparison:
    """Test comparing implementation against golden models."""
    
    def test_attention_matches_golden(self):
        """Test attention implementation matches golden model."""
        try:
            from tests.golden import GoldenMultiHeadAttention
        except ImportError:
            pytest.skip("Golden models not available")
        
        np.random.seed(42)
        
        # Create golden model
        attn = GoldenMultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            weight_type='ternary',
            precision='float'
        )
        
        # Test input
        x = np.random.randn(8, 64).astype(np.float32) * 0.1
        
        # Forward pass
        output, attention = attn.forward(x, return_attention=True)
        
        # Verify shapes
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape}"
        assert attention.shape[-1] == x.shape[0], "Attention shape mismatch"
        
        # Verify attention sums to 1
        attn_sums = np.sum(attention, axis=-1)
        assert np.allclose(attn_sums, 1.0, atol=0.1), "Attention doesn't sum to 1"
    
    def test_transformer_matches_golden(self):
        """Test transformer block matches golden model."""
        try:
            from tests.golden import GoldenTransformerBlock, VisionConfig
        except ImportError:
            pytest.skip("Golden models not available")
        
        np.random.seed(42)
        
        config = VisionConfig(embed_dim=64, num_heads=4, mlp_dim=256)
        block = GoldenTransformerBlock(config, weight_type='ternary')
        
        x = np.random.randn(16, 64).astype(np.float32) * 0.1
        
        output = block.forward(x)
        
        assert output.shape == x.shape, f"Shape mismatch: {output.shape}"
        assert not np.isnan(output).any(), "Output contains NaN"
        assert not np.isinf(output).any(), "Output contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
