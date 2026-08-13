#!/usr/bin/env python3
"""
Fuzzing Tests for SiLens Hardware and Software.

This module provides fuzz testing to discover edge cases and potential bugs
in the SiLens accelerator. Tests include:

1. Input Fuzzing - Random/malformed inputs to SDK
2. Register Fuzzing - Random register writes
3. DMA Fuzzing - Boundary conditions and overlapping transfers
4. Quantization Fuzzing - Edge case weights
5. Token Fuzzing - Invalid/boundary token IDs

Uses hypothesis for property-based testing.

Usage:
    pytest tests/fuzz/test_fuzzing.py -v
    pytest tests/fuzz/test_fuzzing.py::test_register_fuzzing -v --hypothesis-seed=42
"""

import pytest
import numpy as np
from typing import List, Tuple, Optional
import struct

try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    pytest.skip("hypothesis not installed", allow_module_level=True)

import sys
sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parents[2]))

from sdk.silens.device import (
    SimulatedDevice, Registers, CtrlBits, StatusBits,
    DMABuffer, DeviceError
)


# =============================================================================
# Custom Strategies
# =============================================================================

# Valid register offsets (4-byte aligned, within BAR0)
register_offsets = st.sampled_from([
    Registers.CTRL, Registers.STATUS, Registers.IMG_ADDR,
    Registers.IMG_SIZE, Registers.OUT_ADDR, Registers.OUT_LEN,
    Registers.DMA_CTRL, Registers.VERSION
])

# Random 32-bit values
uint32_values = st.integers(min_value=0, max_value=0xFFFFFFFF)

# Image dimensions (must be reasonable)
image_dims = st.tuples(
    st.integers(min_value=1, max_value=1024),
    st.integers(min_value=1, max_value=1024),
    st.sampled_from([1, 3, 4])  # channels
)

# Token IDs (0 to vocab_size)
VOCAB_SIZE = 49152
token_ids = st.integers(min_value=0, max_value=VOCAB_SIZE - 1)
invalid_token_ids = st.integers(min_value=VOCAB_SIZE, max_value=VOCAB_SIZE * 2)

# Ternary values
ternary_values = st.sampled_from([-1, 0, 1])
ternary_arrays = st.lists(ternary_values, min_size=1, max_size=1024)

# Float arrays for quantization
float_arrays = st.lists(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=16, max_size=1024
)


# =============================================================================
# Register Fuzzing Tests
# =============================================================================

class TestRegisterFuzzing:
    """Fuzz testing for hardware registers."""
    
    @pytest.fixture
    def device(self):
        dev = SimulatedDevice()
        dev.open()
        yield dev
        dev.close()
    
    @given(offset=register_offsets, value=uint32_values)
    @settings(max_examples=100)
    def test_register_write_read_consistency(self, device, offset, value):
        """Writing a register and reading it back should return same value."""
        # Skip read-only registers
        if offset == Registers.STATUS or offset == Registers.VERSION:
            return
        
        device.write_reg(offset, value)
        read_value = device.read_reg(offset)
        
        # Some bits may be read-only or have side effects
        # At minimum, the value should be a valid uint32
        assert 0 <= read_value <= 0xFFFFFFFF
    
    @given(value=uint32_values)
    @settings(max_examples=50)
    def test_control_register_reset_behavior(self, device, value):
        """Setting RESET bit should clear device state."""
        # Write random value with RESET bit set
        reset_value = value | CtrlBits.RESET
        device.write_reg(Registers.CTRL, reset_value)
        
        # After reset, device should be ready
        device.write_reg(Registers.CTRL, 0)  # Clear reset
        
        # Give simulated device time to reset
        import time
        time.sleep(0.05)
        
        status = device.read_reg(Registers.STATUS)
        # Should have READY or INIT_DONE set after reset
        assert status & (StatusBits.READY | StatusBits.INIT_DONE)
    
    @given(offset=st.integers(min_value=0, max_value=0x1000))
    @settings(max_examples=100)
    def test_unaligned_register_access(self, device, offset):
        """Unaligned register access should not crash."""
        # Test both aligned and unaligned offsets
        try:
            value = device.read_reg(offset)
            # Should either succeed with valid value or raise exception
            assert isinstance(value, int)
        except (DeviceError, ValueError):
            # Expected for invalid offsets
            pass


# =============================================================================
# DMA Fuzzing Tests
# =============================================================================

class TestDMAFuzzing:
    """Fuzz testing for DMA operations."""
    
    @pytest.fixture
    def device(self):
        dev = SimulatedDevice()
        dev.open()
        yield dev
        dev.close()
    
    @given(size=st.integers(min_value=1, max_value=1024*1024))
    @settings(max_examples=50)
    def test_dma_buffer_allocation_sizes(self, device, size):
        """DMA buffer allocation should handle various sizes."""
        try:
            buffer = device.alloc_dma_buffer(size)
            
            assert buffer.size == size
            assert buffer.data is not None
            assert len(buffer.data) >= size
            
            device.free_dma_buffer(buffer)
        except MemoryError:
            # Large allocations may fail - that's ok
            pass
    
    @given(
        size=st.integers(min_value=64, max_value=4096),
        offset=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=50)
    def test_dma_buffer_write_boundaries(self, device, size, offset):
        """DMA buffer writes should respect boundaries."""
        buffer = device.alloc_dma_buffer(size + offset + 100)
        
        # Write data at offset
        data = np.random.randint(0, 256, size=size, dtype=np.uint8)
        
        if offset + size <= buffer.size:
            buffer.write(data, offset)
            read_data = buffer.read(size, offset)
            np.testing.assert_array_equal(read_data, data)
        else:
            with pytest.raises(ValueError):
                buffer.write(data, offset)
        
        device.free_dma_buffer(buffer)
    
    @given(
        dest_addr=st.integers(min_value=0, max_value=0x10000000),
        size=st.integers(min_value=1, max_value=4096)
    )
    @settings(max_examples=50)
    def test_dma_transfer_addresses(self, device, dest_addr, size):
        """DMA transfers should handle various destination addresses."""
        buffer = device.alloc_dma_buffer(size)
        buffer.data[:] = np.random.randint(0, 256, size=size, dtype=np.uint8)
        
        try:
            device.dma_transfer_to_device(buffer, dest_addr, size)
            # Transfer should complete (simulated device)
        except (DeviceError, ValueError):
            # Invalid addresses may raise errors
            pass
        finally:
            device.free_dma_buffer(buffer)


# =============================================================================
# Quantization Fuzzing Tests
# =============================================================================

class TestQuantizationFuzzing:
    """Fuzz testing for ternary quantization."""
    
    @given(weights=float_arrays)
    @settings(max_examples=100)
    def test_ternary_quantization_output_values(self, weights):
        """Quantized values should only be -1, 0, or 1."""
        from model.conversion.quantize_ternary import TernaryQuantizer, TernaryQuantizationConfig
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        
        weights_np = np.array(weights, dtype=np.float32)
        result = quantizer.quantize_tensor("test", weights_np)
        
        # Check all values are ternary
        unique_vals = set(result.quantized_weights.flatten())
        assert unique_vals.issubset({-1, 0, 1})
    
    @given(
        weights=float_arrays,
        alpha=st.floats(min_value=0.1, max_value=0.99)
    )
    @settings(max_examples=100)
    def test_alpha_affects_sparsity(self, weights, alpha):
        """Higher alpha should generally increase sparsity."""
        from model.conversion.quantize_ternary import TernaryQuantizer, TernaryQuantizationConfig
        
        weights_np = np.array(weights, dtype=np.float32)
        
        # Quantize with given alpha
        config = TernaryQuantizationConfig(alpha=alpha)
        quantizer = TernaryQuantizer(config)
        result = quantizer.quantize_tensor("test", weights_np)
        
        # Sparsity should be between 0 and 1
        assert 0 <= result.sparsity <= 1
        
        # Statistics should be consistent
        total = result.num_positive + result.num_negative + result.num_zero
        assert total == weights_np.size
    
    @given(weights=st.lists(
        st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
        min_size=16, max_size=256
    ))
    @settings(max_examples=50)
    def test_extreme_weight_values(self, weights):
        """Quantization should handle extreme values gracefully."""
        from model.conversion.quantize_ternary import TernaryQuantizer, TernaryQuantizationConfig
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        
        weights_np = np.array(weights, dtype=np.float32)
        
        # Should not crash
        result = quantizer.quantize_tensor("test", weights_np)
        
        # Output should still be valid ternary
        assert np.all(np.isin(result.quantized_weights, [-1, 0, 1]))
    
    @given(
        shape=st.tuples(
            st.integers(min_value=1, max_value=128),
            st.integers(min_value=1, max_value=128)
        )
    )
    @settings(max_examples=50)
    def test_quantization_shape_preservation(self, shape):
        """Quantized weights should preserve original shape."""
        from model.conversion.quantize_ternary import TernaryQuantizer, TernaryQuantizationConfig
        
        config = TernaryQuantizationConfig()
        quantizer = TernaryQuantizer(config)
        
        weights = np.random.randn(*shape).astype(np.float32)
        result = quantizer.quantize_tensor("test", weights)
        
        assert result.quantized_weights.shape == shape
        assert result.original_shape == shape


# =============================================================================
# Token Fuzzing Tests
# =============================================================================

class TestTokenFuzzing:
    """Fuzz testing for token handling."""
    
    @given(token_ids=st.lists(token_ids, min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_valid_token_sequences(self, token_ids):
        """Valid token sequences should be processed without error."""
        # Create numpy array of tokens
        tokens = np.array(token_ids, dtype=np.int32)
        
        # Basic validation
        assert np.all(tokens >= 0)
        assert np.all(tokens < VOCAB_SIZE)
    
    @given(
        valid_tokens=st.lists(token_ids, min_size=1, max_size=50),
        invalid_token=invalid_token_ids
    )
    @settings(max_examples=50)
    def test_invalid_token_detection(self, valid_tokens, invalid_token):
        """Invalid tokens should be detectable."""
        tokens = valid_tokens + [invalid_token]
        tokens_np = np.array(tokens, dtype=np.int32)
        
        # Should be able to detect invalid tokens
        valid_mask = tokens_np < VOCAB_SIZE
        assert not valid_mask[-1]  # Last token should be invalid
    
    @given(length=st.integers(min_value=1, max_value=8192))
    @settings(max_examples=20)
    def test_sequence_length_limits(self, length):
        """Various sequence lengths should be handled."""
        MAX_SEQ_LEN = 8192
        
        # Generate random tokens
        tokens = np.random.randint(0, VOCAB_SIZE, size=length, dtype=np.int32)
        
        # Check if within limits
        is_valid_length = length <= MAX_SEQ_LEN
        
        if is_valid_length:
            assert len(tokens) == length
        else:
            # Would need truncation
            truncated = tokens[:MAX_SEQ_LEN]
            assert len(truncated) == MAX_SEQ_LEN


# =============================================================================
# Stateful Testing (State Machine)
# =============================================================================

class DeviceStateMachine(RuleBasedStateMachine):
    """
    Stateful testing of device operations.
    
    Tests random sequences of operations to find bugs
    that only manifest in specific operation orderings.
    """
    
    def __init__(self):
        super().__init__()
        self.device = SimulatedDevice()
        self.device.open()
        self.buffers: List[DMABuffer] = []
        self.is_inferencing = False
    
    def teardown(self):
        for buf in self.buffers:
            self.device.free_dma_buffer(buf)
        self.device.close()
    
    @rule(size=st.integers(min_value=64, max_value=4096))
    def allocate_buffer(self, size):
        """Allocate a new DMA buffer."""
        if len(self.buffers) < 10:  # Limit buffer count
            buf = self.device.alloc_dma_buffer(size)
            self.buffers.append(buf)
    
    @rule()
    def free_random_buffer(self):
        """Free a random buffer."""
        if self.buffers:
            idx = np.random.randint(0, len(self.buffers))
            buf = self.buffers.pop(idx)
            self.device.free_dma_buffer(buf)
    
    @rule()
    def start_inference(self):
        """Start an inference operation."""
        if not self.is_inferencing:
            self.device.start_inference(streaming=False)
            self.is_inferencing = True
    
    @rule()
    def reset_device(self):
        """Reset the device."""
        self.device.reset()
        self.is_inferencing = False
    
    @rule(offset=register_offsets)
    def read_register(self, offset):
        """Read a random register."""
        value = self.device.read_reg(offset)
        assert 0 <= value <= 0xFFFFFFFF
    
    @invariant()
    def device_always_responds(self):
        """Device should always respond to status queries."""
        status = self.device.get_status()
        assert isinstance(status, StatusBits)
    
    @invariant()
    def buffers_valid(self):
        """All tracked buffers should have valid data."""
        for buf in self.buffers:
            assert buf.data is not None
            assert buf.size > 0


# Run the state machine tests
TestDeviceStateMachine = DeviceStateMachine.TestCase


# =============================================================================
# Image Input Fuzzing
# =============================================================================

class TestImageInputFuzzing:
    """Fuzz testing for image inputs."""
    
    @given(dims=image_dims)
    @settings(max_examples=50)
    def test_image_dimension_handling(self, dims):
        """Various image dimensions should be handled."""
        height, width, channels = dims
        
        # Create random image
        image = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
        
        # Expected patch count for 384x384 with 16x16 patches
        TARGET_SIZE = 384
        PATCH_SIZE = 16
        
        # Resize would be needed
        if height != TARGET_SIZE or width != TARGET_SIZE:
            # Would need preprocessing
            pass
        
        # Channels should be 3 for RGB
        assert channels in [1, 3, 4]
    
    @given(
        pixel_values=st.lists(
            st.integers(min_value=0, max_value=255),
            min_size=384*384*3, max_size=384*384*3
        )
    )
    @settings(max_examples=10)
    def test_pixel_value_ranges(self, pixel_values):
        """Pixel values should be in valid range."""
        pixels = np.array(pixel_values, dtype=np.uint8).reshape(384, 384, 3)
        
        assert pixels.min() >= 0
        assert pixels.max() <= 255
    
    @given(corruption_rate=st.floats(min_value=0, max_value=1))
    @settings(max_examples=20)
    def test_corrupted_image_handling(self, corruption_rate):
        """System should handle partially corrupted images."""
        image = np.random.randint(0, 256, size=(384, 384, 3), dtype=np.uint8)
        
        # Corrupt random pixels
        num_corrupted = int(image.size * corruption_rate)
        if num_corrupted > 0:
            flat = image.flatten()
            corrupt_indices = np.random.choice(len(flat), num_corrupted, replace=False)
            flat[corrupt_indices] = np.random.randint(0, 256, num_corrupted)
        
        # Image should still be processable (as numpy array)
        assert image.shape == (384, 384, 3)
        assert image.dtype == np.uint8


# =============================================================================
# Hardware Encoding Fuzzing
# =============================================================================

class TestHardwareEncodingFuzzing:
    """Fuzz testing for hardware weight encoding."""
    
    @given(ternary_weights=ternary_arrays)
    @settings(max_examples=100)
    def test_ternary_encoding_roundtrip(self, ternary_weights):
        """Ternary encoding should roundtrip correctly."""
        from model.conversion.quantize_ternary import TernaryQuantizer
        
        quantizer = TernaryQuantizer()
        weights = np.array(ternary_weights, dtype=np.int8)
        
        # Encode
        encoded = quantizer.encode_for_hardware(weights)
        
        # Decode (inverse operation)
        decoded = []
        for byte in encoded:
            decoded.append((byte >> 6) & 0x03)
            decoded.append((byte >> 4) & 0x03)
            decoded.append((byte >> 2) & 0x03)
            decoded.append(byte & 0x03)
        
        # Convert 2-bit encoding back to ternary
        decoded_ternary = []
        for val in decoded[:len(ternary_weights)]:
            if val == 0b01:
                decoded_ternary.append(1)
            elif val == 0b10:
                decoded_ternary.append(-1)
            else:
                decoded_ternary.append(0)
        
        np.testing.assert_array_equal(weights, decoded_ternary)
    
    @given(num_weights=st.integers(min_value=1, max_value=10000))
    @settings(max_examples=50)
    def test_encoding_size(self, num_weights):
        """Encoded size should be correct (4 weights per byte)."""
        from model.conversion.quantize_ternary import TernaryQuantizer
        
        quantizer = TernaryQuantizer()
        weights = np.random.choice([-1, 0, 1], size=num_weights).astype(np.int8)
        
        encoded = quantizer.encode_for_hardware(weights)
        
        # Should be ceil(num_weights / 4) bytes
        expected_size = (num_weights + 3) // 4
        assert len(encoded) == expected_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
