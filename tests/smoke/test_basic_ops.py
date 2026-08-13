#!/usr/bin/env python3
"""
SiLens Smoke Test - Basic Operations
=====================================

Quick tests for basic computational operations.
"""

import pytest
import numpy as np


class TestBasicMath:
    """Test basic mathematical operations."""
    
    def test_ternary_encoding(self):
        """Test ternary weight encoding."""
        # Ternary values: -1, 0, +1
        # Encoding: 00=0, 01=+1, 10=-1
        
        values = [-1, 0, 1, 1, 0, -1, 1, -1]
        
        encoded = []
        for v in values:
            if v == 1:
                encoded.append(0b01)
            elif v == -1:
                encoded.append(0b10)
            else:
                encoded.append(0b00)
        
        # Pack into bytes
        packed = 0
        for i, e in enumerate(encoded[:4]):
            packed |= e << (6 - i * 2)
        
        # Unpack
        unpacked = []
        for i in range(4):
            bits = (packed >> (6 - i * 2)) & 0x03
            if bits == 0b01:
                unpacked.append(1)
            elif bits == 0b10:
                unpacked.append(-1)
            else:
                unpacked.append(0)
        
        assert unpacked == values[:4]
    
    def test_fixed_point_conversion(self):
        """Test fixed-point conversion."""
        frac_bits = 4
        scale = 1 << frac_bits
        
        # Float to fixed
        values = [0.0, 0.5, 1.0, -1.0, 1.5, -0.25]
        fixed = [int(round(v * scale)) for v in values]
        
        # Fixed to float
        recovered = [f / scale for f in fixed]
        
        # Check within quantization error
        for orig, rec in zip(values, recovered):
            assert abs(orig - rec) <= 0.5 / scale
    
    def test_popcount(self):
        """Test population count."""
        test_cases = [
            (0b0000, 0),
            (0b0001, 1),
            (0b0011, 2),
            (0b0111, 3),
            (0b1111, 4),
            (0b10101010, 4),
            (0b11111111, 8),
        ]
        
        for value, expected in test_cases:
            actual = bin(value).count('1')
            assert actual == expected, f"popcount({value:08b}) = {actual}, expected {expected}"
    
    def test_ternary_matmul(self):
        """Test ternary matrix multiplication."""
        np.random.seed(42)
        
        # Create ternary weights
        weights = np.random.choice([-1, 0, 1], size=(4, 8), 
                                   p=[0.35, 0.3, 0.35]).astype(np.int8)
        
        # Create input
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        
        # Reference matmul
        expected = x @ weights.astype(np.float32).T
        
        # Ternary matmul
        pos_mask = (weights == 1)
        neg_mask = (weights == -1)
        
        actual = np.zeros(4, dtype=np.float32)
        for i in range(4):
            actual[i] = np.sum(x[pos_mask[i]]) - np.sum(x[neg_mask[i]])
        
        assert np.allclose(expected, actual)


class TestNeuralOps:
    """Test neural network operations."""
    
    def test_softmax(self):
        """Test softmax computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Stable softmax
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        softmax = exp_x / np.sum(exp_x)
        
        # Properties
        assert np.isclose(np.sum(softmax), 1.0)
        assert np.all(softmax >= 0)
        assert np.all(softmax <= 1)
        assert softmax[3] > softmax[0]  # Larger input -> larger output
    
    def test_gelu(self):
        """Test GELU activation."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # Exact GELU
        gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        
        # Properties
        assert np.isclose(gelu[2], 0.0)  # GELU(0) = 0
        assert gelu[3] > 0  # GELU(1) > 0
        assert gelu[4] > gelu[3]  # Monotonic for positive
    
    def test_silu(self):
        """Test SiLU (Swish) activation."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # SiLU = x * sigmoid(x)
        sigmoid = 1 / (1 + np.exp(-x))
        silu = x * sigmoid
        
        # Properties
        assert np.isclose(silu[2], 0.0)  # SiLU(0) = 0
        assert silu[0] < 0  # Negative for negative input
        assert silu[4] > silu[3]  # Monotonic for positive
    
    def test_layer_norm(self):
        """Test layer normalization."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # LayerNorm
        mean = np.mean(x)
        var = np.var(x)
        eps = 1e-6
        normalized = (x - mean) / np.sqrt(var + eps)
        
        # Properties
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-6)
        assert np.isclose(np.std(normalized), 1.0, atol=1e-6)
    
    def test_rms_norm(self):
        """Test RMS normalization."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # RMSNorm
        rms = np.sqrt(np.mean(x**2) + 1e-6)
        normalized = x / rms
        
        # Properties: output RMS should be ~1
        output_rms = np.sqrt(np.mean(normalized**2))
        assert np.isclose(output_rms, 1.0, atol=0.01)


class TestArrayOperations:
    """Test array manipulation operations."""
    
    def test_reshape_for_attention(self):
        """Test reshape for multi-head attention."""
        batch = 2
        seq_len = 8
        embed_dim = 64
        num_heads = 4
        head_dim = embed_dim // num_heads
        
        # Input: [batch, seq_len, embed_dim]
        x = np.random.randn(batch, seq_len, embed_dim)
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        x_reshaped = x.reshape(batch, seq_len, num_heads, head_dim)
        
        # Transpose to [batch, num_heads, seq_len, head_dim]
        x_transposed = x_reshaped.transpose(0, 2, 1, 3)
        
        assert x_transposed.shape == (batch, num_heads, seq_len, head_dim)
    
    def test_patchify_image(self):
        """Test image patchification."""
        image_size = 56
        patch_size = 14
        channels = 3
        
        # Create image
        image = np.random.rand(image_size, image_size, channels)
        
        # Patchify
        num_patches_h = image_size // patch_size
        num_patches_w = image_size // patch_size
        
        patches = image.reshape(
            num_patches_h, patch_size,
            num_patches_w, patch_size,
            channels
        )
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(
            num_patches_h * num_patches_w,
            patch_size * patch_size * channels
        )
        
        expected_num_patches = (image_size // patch_size) ** 2
        expected_patch_dim = patch_size * patch_size * channels
        
        assert patches.shape == (expected_num_patches, expected_patch_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
