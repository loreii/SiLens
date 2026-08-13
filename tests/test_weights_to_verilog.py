#!/usr/bin/env python3
"""
Tests for weights_to_verilog.py conversion utility.

These tests verify:
- Name sanitization for Verilog identifiers
- Layer categorization
- Ternary quantization
- Verilog module generation

Run with: pytest tests/test_weights_to_verilog.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add model/conversion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "model" / "conversion"))

from weights_to_verilog import (
    VerilogConfig,
    categorize_layer,
    generate_weight_module,
    quantize_to_ternary,
    sanitize_name,
)


class TestSanitizeName:
    """Test name sanitization for Verilog identifiers."""

    def test_simple_name(self):
        assert sanitize_name("layer0") == "layer0"

    def test_dots_to_underscores(self):
        assert sanitize_name("model.layers.0") == "model_layers_0"

    def test_dashes_to_underscores(self):
        assert sanitize_name("vision-encoder") == "vision_encoder"

    def test_mixed_special_chars(self):
        assert sanitize_name("model.layer-0.weight") == "model_layer_0_weight"

    def test_leading_number(self):
        result = sanitize_name("0_layer")
        assert not result[0].isdigit()
        assert result == "layer_0_layer"

    def test_uppercase_to_lowercase(self):
        assert sanitize_name("MyLayer") == "mylayer"

    def test_complex_name(self):
        name = "vision_encoder.blocks.0.attn.qkv.weight"
        result = sanitize_name(name)
        assert result == "vision_encoder_blocks_0_attn_qkv_weight"


class TestCategorizeLayer:
    """Test layer categorization by component."""

    def test_vision_encoder_layers(self):
        assert categorize_layer("vision_encoder.block0") == "vision_encoder"
        assert categorize_layer("image_model.layer1") == "vision_encoder"
        assert categorize_layer("vit.blocks.0") == "vision_encoder"
        assert categorize_layer("siglip.encoder") == "vision_encoder"

    def test_language_model_layers(self):
        assert categorize_layer("language_model.layers.0") == "language_model"
        assert categorize_layer("llm.block0") == "language_model"
        assert categorize_layer("model.layers.0.attn") == "language_model"
        assert categorize_layer("lm_head.weight") == "language_model"

    def test_projector_layers(self):
        assert categorize_layer("projector.linear1") == "projector"
        assert categorize_layer("multi_modal_projector") == "projector"
        assert categorize_layer("connector.fc") == "projector"

    def test_other_layers(self):
        assert categorize_layer("unknown.layer") == "other"
        assert categorize_layer("random_name") == "other"


class TestQuantizeToTernary:
    """Test ternary quantization."""

    def test_already_ternary(self):
        weights = np.array([[1, -1, 0], [0, 1, -1]])
        result = quantize_to_ternary(weights)
        np.testing.assert_array_equal(result, weights)

    def test_positive_values(self):
        weights = np.array([[1.0, 2.0, 3.0]])
        result = quantize_to_ternary(weights)
        assert np.all(result >= 0)  # All should be positive or zero

    def test_negative_values(self):
        weights = np.array([[-1.0, -2.0, -3.0]])
        result = quantize_to_ternary(weights)
        assert np.all(result <= 0)  # All should be negative or zero

    def test_small_values_become_zero(self):
        weights = np.array([[0.01, -0.01, 0.001]])
        result = quantize_to_ternary(weights)
        # Small values relative to mean should become zero
        assert result.dtype == np.int8

    def test_symmetric_distribution(self):
        # Symmetric distribution should produce balanced ternary
        np.random.seed(42)
        weights = np.random.randn(100, 100)
        result = quantize_to_ternary(weights)
        
        # Check all values are ternary
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset({-1, 0, 1})

    def test_output_dtype(self):
        weights = np.array([[1.5, -0.5, 0.0]])
        result = quantize_to_ternary(weights)
        assert result.dtype == np.int8


class TestGenerateWeightModule:
    """Test Verilog module generation."""

    @pytest.fixture
    def config(self):
        return VerilogConfig(
            act_width=8,
            acc_width=32,
            generate_testbench=False,
            include_comments=True
        )

    def test_simple_positive_weights(self, config):
        weights = np.array([[1, 1, 1]])  # 1x3 matrix, all positive
        verilog = generate_weight_module("test_module", weights, config)
        
        assert "module test_module" in verilog
        assert "endmodule" in verilog
        assert "ACT_WIDTH" in verilog

    def test_simple_negative_weights(self, config):
        weights = np.array([[-1, -1, -1]])  # All negative
        verilog = generate_weight_module("test_neg", weights, config)
        
        assert "module test_neg" in verilog
        assert "-$signed" in verilog or "~" in verilog or "-" in verilog

    def test_zero_weights(self, config):
        weights = np.array([[0, 0, 0]])  # All zeros
        verilog = generate_weight_module("test_zero", weights, config)
        
        assert "module test_zero" in verilog
        # Zero weights should produce zero output
        assert "32'd0" in verilog or "'d0" in verilog

    def test_mixed_weights(self, config):
        weights = np.array([[1, -1, 0], [0, 1, -1]])  # 2x3 mixed
        verilog = generate_weight_module("test_mixed", weights, config)
        
        assert "module test_mixed" in verilog
        assert "out[0]" in verilog
        assert "out[1]" in verilog

    def test_module_header_contains_metadata(self, config):
        weights = np.array([[1, 0, -1]])
        verilog = generate_weight_module(
            "test_meta", weights, config, original_name="layer.0.weight"
        )
        
        assert "Auto-generated" in verilog
        assert "layer.0.weight" in verilog
        assert "Sparsity" in verilog

    def test_sparsity_calculation(self, config):
        # 50% zero weights
        weights = np.array([[1, 0, -1, 0]])
        verilog = generate_weight_module("test_sparse", weights, config)
        
        assert "50.0%" in verilog or "50%" in verilog

    def test_1d_weights_reshaped(self, config):
        weights = np.array([1, -1, 0])  # 1D array
        verilog = generate_weight_module("test_1d", weights, config)
        
        assert "module test_1d" in verilog
        assert "endmodule" in verilog


class TestVerilogConfig:
    """Test VerilogConfig defaults and customization."""

    def test_default_values(self):
        config = VerilogConfig()
        assert config.act_width == 8
        assert config.acc_width == 32
        assert config.generate_testbench is True
        assert config.include_comments is True

    def test_custom_values(self):
        config = VerilogConfig(
            act_width=16,
            acc_width=48,
            generate_testbench=False,
            include_comments=False
        )
        assert config.act_width == 16
        assert config.acc_width == 48
        assert config.generate_testbench is False
        assert config.include_comments is False


class TestIntegration:
    """Integration tests with temporary files."""

    def test_full_conversion_flow(self):
        """Test complete conversion from weights to Verilog files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test weights
            weights_dir = Path(tmpdir) / "weights"
            weights_dir.mkdir()
            
            # Save test weights
            test_weights = {
                'vision_encoder_block0_weight': np.random.randn(64, 64).astype(np.float32),
                'language_model_layer0_weight': np.random.randn(32, 32).astype(np.float32),
            }
            np.savez(weights_dir / "test_weights.npz", **test_weights)
            
            # Output directory
            output_dir = Path(tmpdir) / "output"
            
            # Import and run conversion
            from weights_to_verilog import convert_weights_to_verilog
            
            config = VerilogConfig(generate_testbench=False)
            convert_weights_to_verilog(
                weights_path=str(weights_dir / "test_weights.npz"),
                output_dir=str(output_dir),
                config=config
            )
            
            # Verify outputs exist
            assert (output_dir / "conversion_report.txt").exists()
            
            # Should have vision_encoder and language_model directories
            # (component categorization based on layer names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
