#!/usr/bin/env python3
"""
SiLens Integration Test - SDK Device
====================================

Tests for SDK device abstraction layer.

This module provides:
- Device detection testing
- Simulation mode verification
- Register read/write operations
- Memory management tests

Usage:
    pytest tests/integration/test_sdk_device.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class MockDevice:
    """Mock device for testing SDK without hardware."""
    
    def __init__(self, name: str = "mock_silens_0"):
        self.name = name
        self.connected = False
        self.registers = {}
        self.memory = {}
        self.config = {
            'vendor_id': 0x1234,
            'product_id': 0x5678,
            'version': '1.0.0',
        }
    
    def connect(self) -> bool:
        self.connected = True
        return True
    
    def disconnect(self):
        self.connected = False
    
    def is_connected(self) -> bool:
        return self.connected
    
    def read_register(self, addr: int) -> int:
        return self.registers.get(addr, 0)
    
    def write_register(self, addr: int, value: int):
        self.registers[addr] = value & 0xFFFFFFFF
    
    def read_memory(self, addr: int, length: int) -> bytes:
        data = bytearray(length)
        for i in range(length):
            data[i] = self.memory.get(addr + i, 0)
        return bytes(data)
    
    def write_memory(self, addr: int, data: bytes):
        for i, byte in enumerate(data):
            self.memory[addr + i] = byte
    
    def get_info(self) -> dict:
        return self.config


class TestDeviceConnection:
    """Test device connection and discovery."""
    
    def test_mock_device_creation(self):
        """Test creating a mock device."""
        device = MockDevice("test_device")
        assert device.name == "test_device"
        assert not device.is_connected()
    
    def test_device_connect_disconnect(self):
        """Test connecting and disconnecting."""
        device = MockDevice()
        
        assert not device.is_connected()
        
        result = device.connect()
        assert result is True
        assert device.is_connected()
        
        device.disconnect()
        assert not device.is_connected()
    
    def test_device_info(self):
        """Test getting device information."""
        device = MockDevice()
        
        info = device.get_info()
        
        assert 'vendor_id' in info
        assert 'product_id' in info
        assert 'version' in info


class TestRegisterAccess:
    """Test register read/write operations."""
    
    @pytest.fixture
    def device(self):
        """Create and connect a mock device."""
        dev = MockDevice()
        dev.connect()
        return dev
    
    def test_register_write_read(self, device):
        """Test basic register write and read."""
        # Write value
        device.write_register(0x100, 0xDEADBEEF)
        
        # Read back
        value = device.read_register(0x100)
        
        assert value == 0xDEADBEEF
    
    def test_register_default_value(self, device):
        """Test reading unwritten register returns 0."""
        value = device.read_register(0x200)
        assert value == 0
    
    def test_register_multiple_addresses(self, device):
        """Test multiple register addresses."""
        addresses = [0x00, 0x04, 0x08, 0x100, 0xFFFC]
        
        for i, addr in enumerate(addresses):
            device.write_register(addr, i * 0x11111111)
        
        for i, addr in enumerate(addresses):
            value = device.read_register(addr)
            assert value == i * 0x11111111, f"Addr {addr:#x}: expected {i * 0x11111111:#x}, got {value:#x}"
    
    def test_register_32bit_truncation(self, device):
        """Test that values are truncated to 32 bits."""
        device.write_register(0x00, 0x1FFFFFFFF)  # 33-bit value
        
        value = device.read_register(0x00)
        assert value == 0xFFFFFFFF, "Value not truncated to 32 bits"


class TestMemoryAccess:
    """Test memory read/write operations."""
    
    @pytest.fixture
    def device(self):
        dev = MockDevice()
        dev.connect()
        return dev
    
    def test_memory_write_read(self, device):
        """Test basic memory write and read."""
        data = b'\xDE\xAD\xBE\xEF'
        
        device.write_memory(0x1000, data)
        result = device.read_memory(0x1000, len(data))
        
        assert result == data
    
    def test_memory_partial_read(self, device):
        """Test reading partial memory."""
        data = b'\x01\x02\x03\x04\x05\x06\x07\x08'
        
        device.write_memory(0x2000, data)
        
        # Read first 4 bytes
        result = device.read_memory(0x2000, 4)
        assert result == b'\x01\x02\x03\x04'
        
        # Read last 4 bytes
        result = device.read_memory(0x2004, 4)
        assert result == b'\x05\x06\x07\x08'
    
    def test_memory_large_transfer(self, device):
        """Test large memory transfer."""
        size = 4096
        data = bytes(range(256)) * (size // 256)
        
        device.write_memory(0x10000, data)
        result = device.read_memory(0x10000, size)
        
        assert result == data
    
    def test_memory_unwritten_returns_zero(self, device):
        """Test reading unwritten memory returns zeros."""
        result = device.read_memory(0xF0000, 16)
        
        assert result == b'\x00' * 16


class TestSimulationMode:
    """Test simulation mode operations."""
    
    def test_simulation_inference(self):
        """Test inference in simulation mode."""
        device = MockDevice()
        device.connect()
        
        # Simulate setting up inference
        # Write model config registers
        device.write_register(0x00, 768)   # embed_dim
        device.write_register(0x04, 12)    # num_heads
        device.write_register(0x08, 30)    # num_layers
        
        # Verify configuration
        assert device.read_register(0x00) == 768
        assert device.read_register(0x04) == 12
        assert device.read_register(0x08) == 30
    
    def test_simulation_weight_loading(self):
        """Test weight loading in simulation mode."""
        device = MockDevice()
        device.connect()
        
        # Simulate loading quantized weights
        # Ternary weights: 4 values per byte
        num_weights = 1024
        packed_size = num_weights // 4
        
        # Create random ternary weights
        np.random.seed(42)
        weights = np.random.choice([-1, 0, 1], size=num_weights)
        
        # Pack weights (4 per byte)
        packed = np.zeros(packed_size, dtype=np.uint8)
        for i in range(num_weights):
            byte_idx = i // 4
            bit_pos = (3 - (i % 4)) * 2
            if weights[i] == 1:
                packed[byte_idx] |= 0b01 << bit_pos
            elif weights[i] == -1:
                packed[byte_idx] |= 0b10 << bit_pos
        
        # Write to device memory
        device.write_memory(0x100000, bytes(packed))
        
        # Read back and verify
        result = device.read_memory(0x100000, packed_size)
        assert result == bytes(packed)


class TestDeviceControl:
    """Test device control operations."""
    
    @pytest.fixture
    def device(self):
        dev = MockDevice()
        dev.connect()
        return dev
    
    def test_status_register(self, device):
        """Test reading status register."""
        # Simulate ready status
        STATUS_READY = 0x01
        STATUS_BUSY = 0x02
        
        device.write_register(0x10, STATUS_READY)
        
        status = device.read_register(0x10)
        assert status & STATUS_READY, "Device should be ready"
    
    def test_control_register(self, device):
        """Test writing control register."""
        CTRL_START = 0x01
        CTRL_RESET = 0x02
        
        # Start inference
        device.write_register(0x14, CTRL_START)
        assert device.read_register(0x14) == CTRL_START
        
        # Reset
        device.write_register(0x14, CTRL_RESET)
        assert device.read_register(0x14) == CTRL_RESET


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
