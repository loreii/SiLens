#!/usr/bin/env python3
"""
SiLens Integration Test - Quantization Accuracy
================================================

Tests quantization accuracy and error bounds.

Usage:
    pytest tests/integration/test_quantization_accuracy.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTernaryQuantization:
    """Test ternary quantization accuracy."""
    
    def test_quantization_basic(self):
        """Test basic ternary quantization."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer, TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        
        # Create test weights
        np.random.seed(42)
        weights = np.random.randn(100, 100).astype(np.float32) * 0.1
        
        result = quantizer.quantize_tensor("test", weights)
        
        # Verify output is ternary
        unique_vals = np.unique(result.quantized_weights)
        assert set(unique_vals).issubset({-1, 0, 1}), \
            f"Non-ternary values found: {unique_vals}"
        
        # Verify error is reasonable
        assert result.mean_abs_error < 0.2, \
            f"Error too high: {result.mean_abs_error}"
    
    def test_quantization_sparsity(self):
        """Test that sparsity is in expected range."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer, TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        np.random.seed(43)
        weights = np.random.randn(1000, 1000).astype(np.float32) * 0.1
        
        for alpha in [0.5, 0.6, 0.7, 0.8]:
            config = TernaryQuantizationConfig(alpha=alpha)
            quantizer = TernaryQuantizer(config)
            result = quantizer.quantize_tensor("test", weights)
            
            # Sparsity should be in reasonable range
            assert 0.05 <= result.sparsity <= 0.6, \
                f"alpha={alpha}: sparsity {result.sparsity:.1%} out of range"
    
    def test_quantization_symmetry(self):
        """Test that quantization is symmetric."""
        try:
            from model.conversion.quantize_ternary import (
                TernaryQuantizer, TernaryQuantizationConfig
            )
        except ImportError:
            pytest.skip("Quantization module not available")
        
        np.random.seed(44)
        weights = np.random.randn(1000, 1000).astype(np.float32) * 0.1
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        result = quantizer.quantize_tensor("test", weights)
        
        # Check symmetry: roughly equal +1 and -1
        pos_ratio = result.num_positive / weights.size
        neg_ratio = result.num_negative / weights.size
        
        assert abs(pos_ratio - neg_ratio) < 0.1, \
            f"Distribution not symmetric: +1={pos_ratio:.1%}, -1={neg_ratio:.1%}"


class TestQuantizationAgainstGolden:
    """Test quantized operations against golden models."""
    
    def test_ternary_matmul_accuracy(self):
        """Test ternary matmul accuracy."""
        np.random.seed(45)
        
        # Create ternary weights
        weights = np.random.choice([-1, 0, 1], 
            size=(64, 128), p=[0.35, 0.3, 0.35]).astype(np.int8)
        
        # Create input
        x = np.random.randn(16, 128).astype(np.float32) * 0.1
        
        # Float matmul reference
        float_result = x @ weights.astype(np.float32).T
        
        # Ternary matmul (simulation)
        pos_mask = (weights == 1)
        neg_mask = (weights == -1)
        
        ternary_result = np.zeros((16, 64), dtype=np.float32)
        for i in range(64):
            pos_sum = np.sum(x[:, pos_mask[i]], axis=-1)
            neg_sum = np.sum(x[:, neg_mask[i]], axis=-1)
            ternary_result[:, i] = pos_sum - neg_sum
        
        # Results should match exactly for ternary weights
        assert np.allclose(float_result, ternary_result), \
            "Ternary matmul doesn't match float matmul"
    
    def test_binary_matmul_accuracy(self):
        """Test binary matmul accuracy."""
        np.random.seed(46)
        
        # Create binary weights {-1, +1}
        weights = np.random.choice([-1, 1], size=(64, 128)).astype(np.int8)
        
        # Create input
        x = np.random.randn(16, 128).astype(np.float32) * 0.1
        
        # Float matmul reference
        float_result = x @ weights.astype(np.float32).T
        
        # Binary matmul via XNOR+popcount (simulation)
        # Binarize input: sign(x)
        x_bin = np.sign(x)
        x_bin[x_bin == 0] = 1  # Treat 0 as +1
        
        binary_result = x_bin @ weights.astype(np.float32).T
        
        # Binary approximation should have some correlation
        correlation = np.corrcoef(float_result.flatten(), binary_result.flatten())[0, 1]
        assert correlation > 0.5, f"Low correlation: {correlation}"
    
    def test_golden_attention_quantized(self):
        """Test golden attention with quantized weights."""
        try:
            from tests.golden.golden_attention import GoldenMultiHeadAttention
        except ImportError:
            pytest.skip("Golden models not available")
        
        # Create attention modules with different weight types
        attn_ternary = GoldenMultiHeadAttention(
            embed_dim=64, num_heads=4, weight_type='ternary'
        )
        attn_float = GoldenMultiHeadAttention(
            embed_dim=64, num_heads=4, weight_type='float'
        )
        
        # Same input
        np.random.seed(47)
        x = np.random.randn(8, 64).astype(np.float32) * 0.1
        
        out_ternary = attn_ternary.forward(x)
        out_float = attn_float.forward(x)
        
        # Outputs should have similar statistics
        assert abs(out_ternary.mean() - out_float.mean()) < 0.5
        assert abs(out_ternary.std() - out_float.std()) < 0.5


class TestFixedPointAccuracy:
    """Test fixed-point arithmetic accuracy."""
    
    def test_fixed_point_roundtrip(self):
        """Test fixed-point conversion roundtrip."""
        try:
            from tests.golden.golden_attention import FixedPointOps
        except ImportError:
            pytest.skip("FixedPointOps not available")
        
        fp = FixedPointOps(width=8, frac_bits=4)
        
        # Test values
        values = np.array([-7.5, -1.0, 0.0, 0.5, 1.0, 7.5], dtype=np.float32)
        
        # Convert to fixed and back
        fixed = fp.float_to_fixed(values)
        recovered = fp.fixed_to_float(fixed)
        
        # Should match within quantization error
        max_error = 1.0 / (1 << fp.frac_bits)  # 0.0625 for 4 frac bits
        assert np.max(np.abs(values - recovered)) <= max_error * 1.5
    
    def test_fixed_point_saturation(self):
        """Test fixed-point saturation."""
        try:
            from tests.golden.golden_attention import FixedPointOps
        except ImportError:
            pytest.skip("FixedPointOps not available")
        
        fp = FixedPointOps(width=8, frac_bits=4)
        
        # Test overflow values
        large_values = np.array([100.0, -100.0], dtype=np.float32)
        
        fixed = fp.float_to_fixed(large_values)
        
        # Should be saturated to max/min
        assert fixed[0] == fp.max_val
        assert fixed[1] == fp.min_val
    
    def test_fixed_point_multiply(self):
        """Test fixed-point multiplication."""
        try:
            from tests.golden.golden_attention import FixedPointOps
        except ImportError:
            pytest.skip("FixedPointOps not available")
        
        fp = FixedPointOps(width=8, frac_bits=4)
        
        # Test multiplication
        a = np.array([1.0, 2.0, -1.5], dtype=np.float32)
        b = np.array([2.0, 0.5, 2.0], dtype=np.float32)
        
        a_fixed = fp.float_to_fixed(a)
        b_fixed = fp.float_to_fixed(b)
        
        result_fixed = fp.mul(a_fixed, b_fixed)
        result_float = fp.fixed_to_float(result_fixed)
        
        expected = a * b
        
        # Check within quantization tolerance
        assert np.allclose(result_float, expected, atol=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
