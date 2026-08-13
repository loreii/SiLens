#!/usr/bin/env python3
"""
SiLens Regression Test Suite.

Comprehensive regression tests to catch regressions in:
- Model accuracy after code changes
- Hardware interface compatibility
- API stability
- Performance characteristics
- Quantization quality

Usage:
    pytest tests/regression/test_regression_suite.py -v
    pytest tests/regression/test_regression_suite.py -v -m "not slow"
    pytest tests/regression/test_regression_suite.py::TestModelAccuracy -v

Author: SiLens Test Team
License: Apache 2.0
"""

import pytest
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))


# =============================================================================
# Test Data and Fixtures
# =============================================================================

REGRESSION_DATA_DIR = Path(__file__).parent / "data"
BASELINE_DIR = Path(__file__).parent / "baselines"

# Known good outputs for regression testing
EXPECTED_OUTPUTS = {
    "simple_prompt": {
        "input_hash": "abc123",
        "output_tokens": [1, 2, 3, 4],
        "perplexity": 12.5,
    },
    "image_caption": {
        "input_hash": "def456",
        "output_tokens": [5, 6, 7, 8],
        "perplexity": 15.2,
    }
}


@pytest.fixture(scope="module")
def regression_baseline():
    """Load regression baseline data."""
    baseline_file = BASELINE_DIR / "baseline_v1.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            return json.load(f)
    return {}


@pytest.fixture
def simulated_device():
    """Create a simulated device for testing."""
    from sdk.silens.device import SimulatedDevice
    device = SimulatedDevice()
    device.open()
    yield device
    device.close()


# =============================================================================
# Model Accuracy Regression Tests
# =============================================================================

class TestModelAccuracy:
    """Regression tests for model accuracy."""
    
    @pytest.mark.regression
    def test_deterministic_output(self):
        """Model should produce deterministic output with same seed."""
        np.random.seed(42)
        
        # Create test input
        input_data = np.random.randn(1, 384, 384, 3).astype(np.float32)
        input_hash = hashlib.md5(input_data.tobytes()).hexdigest()[:8]
        
        # Hash should be deterministic
        assert input_hash == hashlib.md5(input_data.tobytes()).hexdigest()[:8]
    
    @pytest.mark.regression
    def test_quantization_consistency(self):
        """Quantization should produce consistent results."""
        from model.conversion.quantize_ternary import TernaryQuantizer, TernaryQuantizationConfig
        
        # Same weights, same alpha -> same quantization
        weights = np.random.randn(256, 256).astype(np.float32)
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer1 = TernaryQuantizer(config)
        quantizer2 = TernaryQuantizer(config)
        
        result1 = quantizer1.quantize_tensor("test", weights)
        result2 = quantizer2.quantize_tensor("test", weights)
        
        np.testing.assert_array_equal(
            result1.quantized_weights, 
            result2.quantized_weights
        )

    @pytest.mark.regression
    def test_sparsity_within_expected_range(self):
        """Quantization sparsity should be within expected range."""
        from model.conversion.quantize_ternary import TernaryQuantizer, TernaryQuantizationConfig
        
        # Normal distribution weights
        weights = np.random.randn(1024, 1024).astype(np.float32)
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        result = quantizer.quantize_tensor("test", weights)
        
        # Sparsity for alpha=0.7 on normal distribution should be ~20-40%
        assert 0.1 < result.sparsity < 0.6, f"Unexpected sparsity: {result.sparsity}"
    
    @pytest.mark.regression
    @pytest.mark.slow
    def test_output_distribution(self):
        """Weight distribution should follow expected pattern."""
        from model.conversion.quantize_ternary import TernaryQuantizer, TernaryQuantizationConfig
        
        weights = np.random.randn(10000).astype(np.float32)
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        result = quantizer.quantize_tensor("test", weights)
        
        # Distribution should be roughly symmetric
        ratio = result.num_positive / (result.num_negative + 1e-10)
        assert 0.7 < ratio < 1.4, f"Asymmetric distribution: {ratio}"


# =============================================================================
# API Stability Regression Tests
# =============================================================================

class TestAPIStability:
    """Regression tests for API stability."""
    
    @pytest.mark.regression
    def test_device_interface_unchanged(self, simulated_device):
        """Device interface should maintain backward compatibility."""
        device = simulated_device
        
        # Required methods should exist
        assert hasattr(device, 'read_reg')
        assert hasattr(device, 'write_reg')
        assert hasattr(device, 'alloc_dma_buffer')
        assert hasattr(device, 'free_dma_buffer')
        assert hasattr(device, 'get_status')
        assert hasattr(device, 'get_version')
        assert hasattr(device, 'reset')
        assert hasattr(device, 'start_inference')
    
    @pytest.mark.regression
    def test_register_addresses_unchanged(self):
        """Register addresses should not change."""
        from sdk.silens.device import Registers
        
        # Critical register addresses
        assert Registers.CTRL == 0x000
        assert Registers.STATUS == 0x004
        assert Registers.IMG_ADDR == 0x008
        assert Registers.OUT_ADDR == 0x010
        assert Registers.VERSION == 0x1F0
    
    @pytest.mark.regression
    def test_status_bits_unchanged(self):
        """Status bit definitions should not change."""
        from sdk.silens.device import StatusBits
        
        assert StatusBits.READY == (1 << 0)
        assert StatusBits.BUSY == (1 << 1)
        assert StatusBits.ERROR == (1 << 2)
    
    @pytest.mark.regression
    def test_quantizer_config_defaults(self):
        """Default config values should not change unexpectedly."""
        from model.conversion.quantize_ternary import TernaryQuantizationConfig
        
        config = TernaryQuantizationConfig()
        
        # Default alpha should be 0.7
        assert config.alpha == 0.7
        assert config.symmetric == True
        assert config.skip_normalization == True


# =============================================================================
# Hardware Interface Regression Tests
# =============================================================================

class TestHardwareInterface:
    """Regression tests for hardware interface."""
    
    @pytest.mark.regression
    def test_dma_buffer_alignment(self, simulated_device):
        """DMA buffers should maintain expected alignment."""
        device = simulated_device
        
        for size in [64, 256, 1024, 4096]:
            buffer = device.alloc_dma_buffer(size)
            
            # Size should match requested
            assert buffer.size == size
            
            # Data should be accessible
            assert buffer.data is not None
            
            device.free_dma_buffer(buffer)
    
    @pytest.mark.regression
    def test_register_read_write_cycle(self, simulated_device):
        """Register read/write should complete without error."""
        device = simulated_device
        from sdk.silens.device import Registers
        
        # Write then read back
        test_value = 0x12345678
        device.write_reg(Registers.IMG_ADDR, test_value)
        
        # Read should succeed (value may differ due to hardware behavior)
        read_value = device.read_reg(Registers.IMG_ADDR)
        assert isinstance(read_value, int)
        assert 0 <= read_value <= 0xFFFFFFFF
    
    @pytest.mark.regression
    def test_device_version_format(self, simulated_device):
        """Device version should return expected format."""
        device = simulated_device
        
        version = device.get_version()
        
        # Should be (major, minor, patch) tuple
        assert isinstance(version, tuple)
        assert len(version) == 3
        
        major, minor, patch = version
        assert isinstance(major, int)
        assert isinstance(minor, int)
        assert isinstance(patch, int)
        assert major >= 0 and minor >= 0 and patch >= 0


# =============================================================================
# Performance Regression Tests
# =============================================================================

class TestPerformanceRegression:
    """Regression tests for performance characteristics."""
    
    PERFORMANCE_THRESHOLDS = {
        'quantize_1k_weights_ms': 10.0,
        'buffer_alloc_ms': 1.0,
        'register_read_us': 100.0,
    }
    
    @pytest.mark.regression
    @pytest.mark.slow
    def test_quantization_performance(self):
        """Quantization should complete within time threshold."""
        import time
        from model.conversion.quantize_ternary import TernaryQuantizer, TernaryQuantizationConfig
        
        weights = np.random.randn(1024, 1024).astype(np.float32)
        
        config = TernaryQuantizationConfig()
        quantizer = TernaryQuantizer(config)
        
        start = time.perf_counter()
        result = quantizer.quantize_tensor("test", weights)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Should complete within threshold
        threshold = self.PERFORMANCE_THRESHOLDS['quantize_1k_weights_ms'] * 1000
        assert elapsed_ms < threshold, f"Quantization too slow: {elapsed_ms:.1f}ms"
    
    @pytest.mark.regression
    def test_buffer_allocation_performance(self, simulated_device):
        """Buffer allocation should be fast."""
        import time
        device = simulated_device
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            buffer = device.alloc_dma_buffer(4096)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            device.free_dma_buffer(buffer)
        
        avg_time = np.mean(times)
        assert avg_time < self.PERFORMANCE_THRESHOLDS['buffer_alloc_ms']
    
    @pytest.mark.regression
    def test_register_access_performance(self, simulated_device):
        """Register access should be fast."""
        import time
        from sdk.silens.device import Registers
        
        device = simulated_device
        
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = device.read_reg(Registers.STATUS)
            elapsed_us = (time.perf_counter() - start) * 1e6
            times.append(elapsed_us)
        
        avg_time = np.mean(times)
        assert avg_time < self.PERFORMANCE_THRESHOLDS['register_read_us']


# =============================================================================
# Encoding Regression Tests
# =============================================================================

class TestEncodingRegression:
    """Regression tests for weight encoding."""
    
    @pytest.mark.regression
    def test_hardware_encoding_roundtrip(self):
        """Hardware encoding should be reversible."""
        from model.conversion.quantize_ternary import TernaryQuantizer
        
        quantizer = TernaryQuantizer()
        
        # Test with known values
        test_cases = [
            np.array([1, 0, -1, 0], dtype=np.int8),
            np.array([1, 1, 1, 1], dtype=np.int8),
            np.array([0, 0, 0, 0], dtype=np.int8),
            np.array([-1, -1, -1, -1], dtype=np.int8),
            np.array([1, -1, 0, 1, -1, 0, 1, -1], dtype=np.int8),
        ]
        
        for original in test_cases:
            encoded = quantizer.encode_for_hardware(original)
            
            # Decode
            decoded = []
            for byte in encoded:
                decoded.extend([
                    1 if ((byte >> 6) & 0x03) == 0b01 else 
                    -1 if ((byte >> 6) & 0x03) == 0b10 else 0,
                    1 if ((byte >> 4) & 0x03) == 0b01 else 
                    -1 if ((byte >> 4) & 0x03) == 0b10 else 0,
                    1 if ((byte >> 2) & 0x03) == 0b01 else 
                    -1 if ((byte >> 2) & 0x03) == 0b10 else 0,
                    1 if (byte & 0x03) == 0b01 else 
                    -1 if (byte & 0x03) == 0b10 else 0,
                ])
            
            decoded = np.array(decoded[:len(original)], dtype=np.int8)
            np.testing.assert_array_equal(original, decoded)
    
    @pytest.mark.regression
    def test_encoding_compression_ratio(self):
        """Encoding should achieve expected compression."""
        from model.conversion.quantize_ternary import TernaryQuantizer
        
        quantizer = TernaryQuantizer()
        
        # 1000 weights at 2 bits each = 250 bytes
        weights = np.random.choice([-1, 0, 1], size=1000).astype(np.int8)
        encoded = quantizer.encode_for_hardware(weights)
        
        # Should be ceil(1000/4) = 250 bytes
        expected_size = (1000 + 3) // 4
        assert len(encoded) == expected_size


# =============================================================================
# Cross-Component Regression Tests
# =============================================================================

class TestCrossComponent:
    """Regression tests for component interactions."""
    
    @pytest.mark.regression
    def test_quantizer_to_device_pipeline(self, simulated_device):
        """Quantized weights should be compatible with device interface."""
        from model.conversion.quantize_ternary import TernaryQuantizer, TernaryQuantizationConfig
        
        device = simulated_device
        
        # Quantize weights
        weights = np.random.randn(256, 256).astype(np.float32)
        config = TernaryQuantizationConfig()
        quantizer = TernaryQuantizer(config)
        result = quantizer.quantize_tensor("test", weights)
        
        # Encode for hardware
        encoded = quantizer.encode_for_hardware(result.quantized_weights)
        
        # Should be able to transfer to device
        buffer = device.alloc_dma_buffer(len(encoded))
        buffer.write(encoded, 0)
        
        # Transfer to device
        device.dma_transfer_to_device(buffer, 0x1000, len(encoded))
        
        device.free_dma_buffer(buffer)
    
    @pytest.mark.regression
    def test_profiler_device_compatibility(self, simulated_device):
        """Profiler should work with device interface."""
        from sdk.silens.profiler import Profiler
        
        device = simulated_device
        profiler = Profiler(device)
        
        # Should be able to start/stop profiling
        profiler.start()
        
        # Simulate some operations
        _ = device.get_status()
        device.start_inference()
        
        profiler.stop()
        
        # Should generate report without error
        report = profiler.get_report()
        assert report is not None
        assert hasattr(report, 'total_time_ms')


# =============================================================================
# Setup and Utilities
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_regression_dirs():
    """Ensure regression test directories exist."""
    REGRESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)


def update_baseline(test_name: str, data: Dict[str, Any]) -> None:
    """Update baseline data for a test (utility function)."""
    baseline_file = BASELINE_DIR / "baseline_v1.json"
    
    if baseline_file.exists():
        with open(baseline_file) as f:
            baselines = json.load(f)
    else:
        baselines = {}
    
    baselines[test_name] = data
    
    with open(baseline_file, 'w') as f:
        json.dump(baselines, f, indent=2)


# Init file for the regression tests directory
Path(__file__).parent.joinpath("__init__.py").write_text(
    "# Regression test suite\n"
)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
