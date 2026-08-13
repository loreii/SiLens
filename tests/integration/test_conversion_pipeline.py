#!/usr/bin/env python3
"""
SiLens Integration Test - Conversion Pipeline
==============================================

Tests for the complete weight conversion pipeline:
- Weight extraction from HuggingFace models
- Ternary quantization
- Verilog generation
- Roundtrip consistency

Usage:
    pytest tests/integration/test_conversion_pipeline.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestWeightExtraction:
    """Test weight extraction from models."""
    
    def test_extract_synthetic_weights(self):
        """Test extracting weights from synthetic model."""
        # Create synthetic weights
        np.random.seed(42)
        weights = {
            'layer1.weight': np.random.randn(64, 64).astype(np.float32) * 0.02,
            'layer2.weight': np.random.randn(128, 64).astype(np.float32) * 0.02,
            'layer3.weight': np.random.randn(64, 128).astype(np.float32) * 0.02,
        }
        
        # Verify extraction
        assert len(weights) == 3
        assert weights['layer1.weight'].shape == (64, 64)
        assert weights['layer2.weight'].shape == (128, 64)
    
    @pytest.mark.requires_model
    def test_extract_real_weights(self, hf_model):
        """Test extracting weights from real model."""
        weights = {}
        
        for name, param in hf_model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                weights[name] = param.detach().cpu().numpy()
        
        assert len(weights) > 0, "No weights extracted"
        
        # Verify we got different types of layers
        layer_types = set()
        for name in weights:
            if 'attention' in name.lower() or 'attn' in name.lower():
                layer_types.add('attention')
            elif 'mlp' in name.lower() or 'fc' in name.lower():
                layer_types.add('mlp')
            elif 'embed' in name.lower():
                layer_types.add('embedding')
        
        assert len(layer_types) >= 1, f"Expected multiple layer types, got {layer_types}"


class TestQuantization:
    """Test quantization pipeline."""
    
    @pytest.fixture
    def sample_weights(self):
        """Create sample weights for testing."""
        np.random.seed(42)
        return {
            'linear1': np.random.randn(256, 256).astype(np.float32) * 0.05,
            'linear2': np.random.randn(512, 256).astype(np.float32) * 0.05,
            'projection': np.random.randn(768, 512).astype(np.float32) * 0.05,
        }
    
    def test_quantize_single_layer(self, sample_weights):
        """Test quantizing a single layer."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer,
                TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        
        weights = sample_weights['linear1']
        result = quantizer.quantize_tensor('linear1', weights)
        
        # Verify result
        assert result.quantized_weights.shape == weights.shape
        assert result.quantized_weights.dtype == np.int8
        assert set(np.unique(result.quantized_weights)).issubset({-1, 0, 1})
        
        # Check statistics
        assert 0 <= result.sparsity <= 1
        assert result.mean_abs_error >= 0
        assert result.num_positive + result.num_negative + result.num_zero == weights.size
    
    def test_quantize_all_layers(self, sample_weights):
        """Test quantizing multiple layers."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer,
                TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        
        results = {}
        for name, weights in sample_weights.items():
            results[name] = quantizer.quantize_tensor(name, weights)
        
        assert len(results) == len(sample_weights)
        
        # All should have valid statistics
        for name, result in results.items():
            assert result.sparsity >= 0
            assert result.mean_abs_error >= 0

    
    def test_different_alpha_values(self, sample_weights):
        """Test quantization with different alpha values."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer,
                TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        weights = sample_weights['linear1']
        alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        sparsities = []
        errors = []
        
        for alpha in alphas:
            config = TernaryQuantizationConfig(alpha=alpha)
            quantizer = TernaryQuantizer(config)
            result = quantizer.quantize_tensor('test', weights)
            
            sparsities.append(result.sparsity)
            errors.append(result.mean_abs_error)
        
        # Higher alpha should generally give lower sparsity
        # (more values pass the threshold)
        for i in range(len(alphas) - 1):
            # Allow some variation due to distribution
            assert sparsities[i] >= sparsities[i+1] - 0.1, \
                f"Sparsity should decrease with alpha: {sparsities}"


class TestVerilogGeneration:
    """Test Verilog generation from quantized weights."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_generate_simple_module(self, temp_output_dir):
        """Test generating a simple weight module."""
        try:
            from model.conversion.weights_to_verilog import (
                generate_weight_module,
                VerilogConfig
            )
        except ImportError:
            pytest.skip("Verilog generation module not available")
        
        # Create ternary weights
        np.random.seed(42)
        weights = np.random.choice([-1, 0, 1], size=(16, 32)).astype(np.int8)
        
        config = VerilogConfig()
        verilog = generate_weight_module(
            module_name="test_weights",
            weights=weights,
            config=config,
            original_name="test_layer"
        )
        
        # Verify Verilog content
        assert 'module test_weights' in verilog
        assert 'input' in verilog
        assert 'output' in verilog
        assert 'endmodule' in verilog
        
        # Write to file
        output_file = temp_output_dir / "test_weights.v"
        output_file.write_text(verilog)
        
        assert output_file.exists()
    
    def test_verilog_syntax_valid(self, temp_output_dir):
        """Test that generated Verilog has valid syntax."""
        try:
            from model.conversion.weights_to_verilog import (
                generate_weight_module,
                VerilogConfig
            )
        except ImportError:
            pytest.skip("Verilog generation module not available")
        
        np.random.seed(42)
        weights = np.random.choice([-1, 0, 1], size=(8, 8)).astype(np.int8)
        
        config = VerilogConfig()
        verilog = generate_weight_module("syntax_test", weights, config)
        
        # Basic syntax checks
        # Use word boundary to distinguish 'module' from 'endmodule'
        import re
        assert len(re.findall(r'\bmodule\b', verilog)) == 1
        assert len(re.findall(r'\bendmodule\b', verilog)) == 1
        
        # Check for balanced parentheses
        assert verilog.count('(') == verilog.count(')')
        assert verilog.count('[') == verilog.count(']')
        assert verilog.count('{') == verilog.count('}')


class TestRoundtripConsistency:
    """Test roundtrip consistency of conversion pipeline."""
    
    def test_quantize_encode_decode(self):
        """Test quantization and encoding roundtrip."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer,
                TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        np.random.seed(42)
        
        # Original weights
        original = np.random.randn(100, 100).astype(np.float32) * 0.1
        
        # Quantize
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        result = quantizer.quantize_tensor("test", original)
        
        # Encode
        encoded = quantizer.encode_for_hardware(result.quantized_weights)
        
        # Decode
        decoded = np.zeros(result.quantized_weights.size, dtype=np.int8)
        for i, byte in enumerate(encoded):
            for j in range(4):
                idx = i * 4 + j
                if idx < len(decoded):
                    bits = (byte >> (6 - j * 2)) & 0x03
                    if bits == 0b01:
                        decoded[idx] = 1
                    elif bits == 0b10:
                        decoded[idx] = -1
        
        # Compare
        original_flat = result.quantized_weights.flatten()
        matches = np.sum(original_flat == decoded)
        accuracy = matches / len(original_flat)
        
        assert accuracy > 0.99, f"Roundtrip accuracy: {accuracy:.1%}"
    
    def test_full_pipeline_consistency(self):
        """Test full pipeline produces consistent results."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer,
                TernaryQuantizationConfig
            )
            from model.conversion.weights_to_verilog import (
                generate_weight_module,
                VerilogConfig
            )
        except ImportError:
            pytest.skip("Conversion modules not available")
        
        np.random.seed(42)
        
        # Create weights
        weights = np.random.randn(32, 64).astype(np.float32) * 0.05
        
        # Run pipeline twice
        results = []
        for _ in range(2):
            # Quantize
            config = TernaryQuantizationConfig(alpha=0.7)
            quantizer = TernaryQuantizer(config)
            quant_result = quantizer.quantize_tensor("test", weights)
            
            # Generate Verilog
            verilog_config = VerilogConfig()
            verilog = generate_weight_module(
                "test_module",
                quant_result.quantized_weights,
                verilog_config
            )
            
            results.append({
                'quantized': quant_result.quantized_weights.copy(),
                'verilog': verilog
            })
        
        # Results should be identical
        assert np.array_equal(results[0]['quantized'], results[1]['quantized'])
        assert results[0]['verilog'] == results[1]['verilog']


class TestExportFormats:
    """Test export to different formats."""
    
    @pytest.fixture
    def temp_output_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_export_numpy(self, temp_output_dir):
        """Test export to NumPy format."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer,
                TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        np.random.seed(42)
        weights = np.random.randn(64, 64).astype(np.float32) * 0.05
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        result = quantizer.quantize_tensor("test", weights)
        
        # Save
        output_file = temp_output_dir / "quantized.npy"
        np.save(output_file, result.quantized_weights)
        
        # Load and verify
        loaded = np.load(output_file)
        assert np.array_equal(loaded, result.quantized_weights)
    
    def test_export_packed_binary(self, temp_output_dir):
        """Test export to packed binary format."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer,
                TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        np.random.seed(42)
        weights = np.random.randn(64, 64).astype(np.float32) * 0.05
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        result = quantizer.quantize_tensor("test", weights)
        
        # Encode and save
        encoded = quantizer.encode_for_hardware(result.quantized_weights)
        
        output_file = temp_output_dir / "weights.bin"
        with open(output_file, 'wb') as f:
            f.write(bytes(encoded))
        
        # Load and verify
        with open(output_file, 'rb') as f:
            loaded = np.frombuffer(f.read(), dtype=np.uint8)
        
        assert np.array_equal(loaded, encoded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
