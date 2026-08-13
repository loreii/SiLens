"""
SiLens Device Abstraction Layer.

Provides unified interface for communicating with SiLens hardware through
various backends (PCIe, USB, or simulation).

Register Map (from hardware spec):
    0x000 CTRL     - Control register
    0x004 STATUS   - Status register
    0x008 IMG_ADDR - Image buffer address
    0x00C IMG_SIZE - Image dimensions
    0x010 OUT_ADDR - Output buffer address
    0x014 OUT_LEN  - Output length
    0x100 DMA_CTRL - DMA control
    0x200+ DEBUG   - Debug registers
"""

from __future__ import annotations

import os
import mmap
import struct
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Optional, List, Callable, Any, Union
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Register Definitions
# =============================================================================

class Registers(IntEnum):
    """SiLens hardware register offsets."""
    CTRL = 0x000
    STATUS = 0x004
    IMG_ADDR = 0x008
    IMG_SIZE = 0x00C
    OUT_ADDR = 0x010
    OUT_LEN = 0x014
    TOKEN_OUT = 0x018
    TOKEN_VALID = 0x01C
    KV_CACHE_ADDR = 0x020
    KV_CACHE_SIZE = 0x024
    INTERRUPT_STATUS = 0x028
    INTERRUPT_ENABLE = 0x02C
    DMA_CTRL = 0x100
    DMA_STATUS = 0x104
    DMA_SRC_ADDR = 0x108
    DMA_DST_ADDR = 0x10C
    DMA_LENGTH = 0x110
    VERSION = 0x1F0
    BUILD_DATE = 0x1F4
    DEBUG_BASE = 0x200


class CtrlBits(IntFlag):
    """Control register bit definitions."""
    ENABLE = 1 << 0
    START_INFERENCE = 1 << 1
    RESET = 1 << 2
    VISION_ENABLE = 1 << 3
    LLM_ENABLE = 1 << 4
    STREAMING_MODE = 1 << 5
    LOW_POWER = 1 << 6


class StatusBits(IntFlag):
    """Status register bit definitions."""
    READY = 1 << 0
    BUSY = 1 << 1
    ERROR = 1 << 2
    VISION_DONE = 1 << 3
    LLM_DONE = 1 << 4
    TOKEN_READY = 1 << 5
    DMA_ACTIVE = 1 << 6
    INIT_DONE = 1 << 7


# Hardware constants
SILENS_VENDOR_ID = 0x1E88  # Placeholder vendor ID
SILENS_DEVICE_ID = 0x0001
BAR0_SIZE = 0x1000  # 4KB for registers
BAR1_SIZE = 0x10000000  # 256MB for DMA buffers

# Image parameters from architecture spec
IMG_SIZE = 384
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 576
VISION_DIM = 768
LLM_DIM = 576
VOCAB_SIZE = 49152
MAX_SEQ_LEN = 8192


# =============================================================================
# Exceptions
# =============================================================================

class DeviceError(Exception):
    """Base exception for device-related errors."""
    pass


class DeviceNotFoundError(DeviceError):
    """Raised when no SiLens device is found."""
    pass


class DeviceTimeoutError(DeviceError):
    """Raised when device operation times out."""
    pass


class DMAError(DeviceError):
    """Raised when DMA operation fails."""
    pass


# =============================================================================
# DMA Buffer Management
# =============================================================================

@dataclass
class DMABuffer:
    """Represents a DMA-capable memory buffer."""
    virtual_addr: int
    physical_addr: int
    size: int
    data: np.ndarray
    _mmap: Optional[mmap.mmap] = field(default=None, repr=False)
    
    def write(self, data: np.ndarray, offset: int = 0) -> None:
        """Write numpy array to buffer at offset."""
        if offset + data.nbytes > self.size:
            raise ValueError(f"Data exceeds buffer size: {offset + data.nbytes} > {self.size}")
        np.copyto(self.data.view(np.uint8)[offset:offset + data.nbytes], 
                  data.view(np.uint8))
    
    def read(self, size: int, offset: int = 0) -> np.ndarray:
        """Read data from buffer at offset."""
        if offset + size > self.size:
            raise ValueError(f"Read exceeds buffer size: {offset + size} > {self.size}")
        return self.data.view(np.uint8)[offset:offset + size].copy()
    
    def zero(self) -> None:
        """Zero the entire buffer."""
        self.data.fill(0)


# =============================================================================
# Abstract Device Interface
# =============================================================================

class SiLensDevice(ABC):
    """
    Abstract base class for SiLens device interfaces.
    
    Provides a unified API for hardware communication regardless of
    the underlying transport (PCIe, USB, or simulation).
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._interrupt_callback: Optional[Callable[[int], None]] = None
        self._is_open = False
    
    @classmethod
    def discover(cls, mode: str = "auto") -> List[SiLensDevice]:
        """
        Discover available SiLens devices.
        
        Args:
            mode: Discovery mode - "auto", "pcie", "usb", or "simulation"
            
        Returns:
            List of discovered devices
            
        Raises:
            DeviceNotFoundError: If no devices found and mode != "simulation"
        """
        devices: List[SiLensDevice] = []
        
        if mode in ("auto", "pcie"):
            devices.extend(PCIeDevice._discover())
        
        if mode in ("auto", "usb"):
            devices.extend(USBDevice._discover())
        
        if mode == "simulation" or (mode == "auto" and not devices):
            logger.info("Using simulated device")
            devices.append(SimulatedDevice())
        
        if not devices and mode != "simulation":
            raise DeviceNotFoundError("No SiLens devices found")
        
        return devices

    
    @abstractmethod
    def open(self) -> None:
        """Open connection to the device."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close connection to the device."""
        pass
    
    @abstractmethod
    def read_reg(self, offset: int) -> int:
        """Read a 32-bit register value."""
        pass
    
    @abstractmethod
    def write_reg(self, offset: int, value: int) -> None:
        """Write a 32-bit register value."""
        pass
    
    @abstractmethod
    def alloc_dma_buffer(self, size: int) -> DMABuffer:
        """Allocate a DMA-capable buffer."""
        pass
    
    @abstractmethod
    def free_dma_buffer(self, buffer: DMABuffer) -> None:
        """Free a DMA buffer."""
        pass
    
    @abstractmethod
    def dma_transfer_to_device(self, buffer: DMABuffer, dest_addr: int, 
                                size: int, offset: int = 0) -> None:
        """Transfer data from host buffer to device memory."""
        pass
    
    @abstractmethod
    def dma_transfer_from_device(self, src_addr: int, buffer: DMABuffer,
                                  size: int, offset: int = 0) -> None:
        """Transfer data from device memory to host buffer."""
        pass
    
    def __enter__(self) -> SiLensDevice:
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    # High-level convenience methods
    
    def get_version(self) -> tuple[int, int, int]:
        """Get hardware version as (major, minor, patch)."""
        version = self.read_reg(Registers.VERSION)
        major = (version >> 16) & 0xFF
        minor = (version >> 8) & 0xFF
        patch = version & 0xFF
        return (major, minor, patch)

    
    def get_status(self) -> StatusBits:
        """Get device status flags."""
        return StatusBits(self.read_reg(Registers.STATUS))
    
    def is_ready(self) -> bool:
        """Check if device is ready for commands."""
        status = self.get_status()
        return bool(status & StatusBits.READY) and not bool(status & StatusBits.BUSY)
    
    def is_busy(self) -> bool:
        """Check if device is currently processing."""
        return bool(self.get_status() & StatusBits.BUSY)
    
    def wait_ready(self, timeout: float = 10.0) -> None:
        """Wait for device to become ready."""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_ready():
                return
            time.sleep(0.001)
        raise DeviceTimeoutError(f"Device not ready after {timeout}s")
    
    def reset(self) -> None:
        """Reset the device."""
        with self._lock:
            self.write_reg(Registers.CTRL, CtrlBits.RESET)
            time.sleep(0.1)
            self.write_reg(Registers.CTRL, 0)
            self.wait_ready()
    
    def start_inference(self, streaming: bool = False) -> None:
        """Start inference execution."""
        ctrl = CtrlBits.ENABLE | CtrlBits.START_INFERENCE
        if streaming:
            ctrl |= CtrlBits.STREAMING_MODE
        self.write_reg(Registers.CTRL, ctrl)
    
    def set_interrupt_callback(self, callback: Optional[Callable[[int], None]]) -> None:
        """Set callback for hardware interrupts."""
        self._interrupt_callback = callback
    
    def enable_interrupts(self, mask: int = 0xFF) -> None:
        """Enable hardware interrupts."""
        self.write_reg(Registers.INTERRUPT_ENABLE, mask)


# =============================================================================
# PCIe Device Implementation
# =============================================================================

class PCIeDevice(SiLensDevice):
    """SiLens device connected via PCIe."""
    
    def __init__(self, pci_slot: str, device_path: str):
        super().__init__()
        self.pci_slot = pci_slot
        self.device_path = device_path
        self._bar0_mmap: Optional[mmap.mmap] = None
        self._bar0_fd: Optional[int] = None
        self._dma_buffers: List[DMABuffer] = []

    
    @staticmethod
    def _discover() -> List[PCIeDevice]:
        """Discover PCIe SiLens devices."""
        devices = []
        pci_path = Path("/sys/bus/pci/devices")
        
        if not pci_path.exists():
            return devices
        
        for slot_path in pci_path.iterdir():
            try:
                vendor = (slot_path / "vendor").read_text().strip()
                device = (slot_path / "device").read_text().strip()
                
                if int(vendor, 16) == SILENS_VENDOR_ID and int(device, 16) == SILENS_DEVICE_ID:
                    # Check for silens driver binding
                    driver_link = slot_path / "driver"
                    if driver_link.exists() and "silens" in os.readlink(str(driver_link)):
                        char_dev = Path(f"/dev/silens{len(devices)}")
                        if char_dev.exists():
                            devices.append(PCIeDevice(slot_path.name, str(char_dev)))
            except (OSError, ValueError):
                continue
        
        return devices
    
    def __repr__(self) -> str:
        return f"PCIeDevice(slot={self.pci_slot})"
    
    def open(self) -> None:
        """Open the PCIe device."""
        if self._is_open:
            return
        
        try:
            # Open character device
            self._bar0_fd = os.open(self.device_path, os.O_RDWR)
            
            # Memory-map BAR0 for register access
            self._bar0_mmap = mmap.mmap(
                self._bar0_fd, 
                BAR0_SIZE,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE
            )
            
            self._is_open = True
            logger.info(f"Opened PCIe device at {self.device_path}")
            
            # Wait for device to be ready
            self.wait_ready(timeout=5.0)
            
        except OSError as e:
            self.close()
            raise DeviceError(f"Failed to open device: {e}")

    
    def close(self) -> None:
        """Close the PCIe device."""
        # Free all DMA buffers
        for buffer in self._dma_buffers[:]:
            self.free_dma_buffer(buffer)
        
        if self._bar0_mmap:
            self._bar0_mmap.close()
            self._bar0_mmap = None
        
        if self._bar0_fd is not None:
            os.close(self._bar0_fd)
            self._bar0_fd = None
        
        self._is_open = False
        logger.info("Closed PCIe device")
    
    def read_reg(self, offset: int) -> int:
        """Read a 32-bit register."""
        if not self._bar0_mmap:
            raise DeviceError("Device not open")
        if offset < 0 or offset >= BAR0_SIZE:
            raise ValueError(f"Invalid register offset: 0x{offset:x}")
        
        self._bar0_mmap.seek(offset)
        data = self._bar0_mmap.read(4)
        return struct.unpack("<I", data)[0]
    
    def write_reg(self, offset: int, value: int) -> None:
        """Write a 32-bit register."""
        if not self._bar0_mmap:
            raise DeviceError("Device not open")
        if offset < 0 or offset >= BAR0_SIZE:
            raise ValueError(f"Invalid register offset: 0x{offset:x}")
        
        self._bar0_mmap.seek(offset)
        self._bar0_mmap.write(struct.pack("<I", value & 0xFFFFFFFF))
    
    def alloc_dma_buffer(self, size: int) -> DMABuffer:
        """Allocate a DMA-capable buffer using the kernel driver."""
        if not self._bar0_fd:
            raise DeviceError("Device not open")
        
        # Use ioctl to allocate DMA buffer via kernel driver
        # For now, we'll use a simplified approach with mmap
        try:
            # Allocate buffer through driver (would use ioctl in practice)
            buf_mmap = mmap.mmap(
                self._bar0_fd,
                size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=BAR0_SIZE  # Start after register region
            )
            
            # Create numpy view of the buffer
            data = np.frombuffer(buf_mmap, dtype=np.uint8)
            
            buffer = DMABuffer(
                virtual_addr=id(buf_mmap),
                physical_addr=0,  # Would be obtained from driver
                size=size,
                data=data,
                _mmap=buf_mmap
            )
            
            self._dma_buffers.append(buffer)
            return buffer
            
        except OSError as e:
            raise DMAError(f"Failed to allocate DMA buffer: {e}")

    
    def free_dma_buffer(self, buffer: DMABuffer) -> None:
        """Free a DMA buffer."""
        if buffer._mmap:
            buffer._mmap.close()
        if buffer in self._dma_buffers:
            self._dma_buffers.remove(buffer)
    
    def dma_transfer_to_device(self, buffer: DMABuffer, dest_addr: int,
                                size: int, offset: int = 0) -> None:
        """Transfer data from host to device using DMA."""
        if not self._is_open:
            raise DeviceError("Device not open")
        
        # Configure DMA engine
        self.write_reg(Registers.DMA_SRC_ADDR, buffer.physical_addr + offset)
        self.write_reg(Registers.DMA_DST_ADDR, dest_addr)
        self.write_reg(Registers.DMA_LENGTH, size)
        self.write_reg(Registers.DMA_CTRL, 0x01)  # Start transfer to device
        
        # Wait for completion
        self._wait_dma_complete()
    
    def dma_transfer_from_device(self, src_addr: int, buffer: DMABuffer,
                                  size: int, offset: int = 0) -> None:
        """Transfer data from device to host using DMA."""
        if not self._is_open:
            raise DeviceError("Device not open")
        
        # Configure DMA engine
        self.write_reg(Registers.DMA_SRC_ADDR, src_addr)
        self.write_reg(Registers.DMA_DST_ADDR, buffer.physical_addr + offset)
        self.write_reg(Registers.DMA_LENGTH, size)
        self.write_reg(Registers.DMA_CTRL, 0x02)  # Start transfer from device
        
        # Wait for completion
        self._wait_dma_complete()
    
    def _wait_dma_complete(self, timeout: float = 5.0) -> None:
        """Wait for DMA transfer to complete."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.read_reg(Registers.DMA_STATUS)
            if status & 0x01:  # Transfer complete
                if status & 0x02:  # Error
                    raise DMAError(f"DMA transfer error: status=0x{status:x}")
                return
            time.sleep(0.0001)
        raise DeviceTimeoutError("DMA transfer timeout")


# =============================================================================
# USB Device Implementation (for FPGA prototypes)
# =============================================================================

class USBDevice(SiLensDevice):
    """SiLens device connected via USB (FPGA prototype)."""
    
    def __init__(self, vid: int, pid: int, serial: Optional[str] = None):
        super().__init__()
        self.vid = vid
        self.pid = pid
        self.serial = serial
        self._usb_dev = None
        self._registers: dict[int, int] = {}
        self._memory: bytearray = bytearray(BAR1_SIZE)

    
    @staticmethod
    def _discover() -> List[USBDevice]:
        """Discover USB SiLens devices."""
        devices = []
        
        try:
            import usb.core
            import usb.util
            
            # Look for SiLens USB devices (FPGA prototype)
            usb_devs = usb.core.find(find_all=True, idVendor=SILENS_VENDOR_ID)
            
            for dev in usb_devs:
                serial = usb.util.get_string(dev, dev.iSerialNumber) if dev.iSerialNumber else None
                devices.append(USBDevice(dev.idVendor, dev.idProduct, serial))
                
        except ImportError:
            logger.debug("pyusb not installed, USB discovery skipped")
        except Exception as e:
            logger.debug(f"USB discovery failed: {e}")
        
        return devices
    
    def __repr__(self) -> str:
        return f"USBDevice(vid=0x{self.vid:04x}, pid=0x{self.pid:04x}, serial={self.serial})"
    
    def open(self) -> None:
        """Open USB connection."""
        if self._is_open:
            return
        
        try:
            import usb.core
            import usb.util
            
            self._usb_dev = usb.core.find(idVendor=self.vid, idProduct=self.pid)
            if self._usb_dev is None:
                raise DeviceNotFoundError("USB device not found")
            
            self._usb_dev.set_configuration()
            self._is_open = True
            logger.info(f"Opened USB device: {self}")
            
        except ImportError:
            raise DeviceError("pyusb is required for USB devices: pip install pyusb")
        except Exception as e:
            raise DeviceError(f"Failed to open USB device: {e}")
    
    def close(self) -> None:
        """Close USB connection."""
        if self._usb_dev:
            try:
                import usb.util
                usb.util.dispose_resources(self._usb_dev)
            except Exception:
                pass
            self._usb_dev = None
        self._is_open = False
        logger.info("Closed USB device")
    
    def read_reg(self, offset: int) -> int:
        """Read register via USB control transfer."""
        if not self._usb_dev:
            raise DeviceError("Device not open")
        
        # USB control transfer for register read
        data = self._usb_dev.ctrl_transfer(
            0xC0,  # bmRequestType: Device-to-host, Vendor, Device
            0x01,  # bRequest: Read register
            offset & 0xFFFF,  # wValue: low 16 bits of offset
            (offset >> 16) & 0xFFFF,  # wIndex: high 16 bits of offset
            4  # wLength: 4 bytes
        )
        return struct.unpack("<I", bytes(data))[0]

    
    def write_reg(self, offset: int, value: int) -> None:
        """Write register via USB control transfer."""
        if not self._usb_dev:
            raise DeviceError("Device not open")
        
        # USB control transfer for register write
        self._usb_dev.ctrl_transfer(
            0x40,  # bmRequestType: Host-to-device, Vendor, Device
            0x02,  # bRequest: Write register
            offset & 0xFFFF,
            (offset >> 16) & 0xFFFF,
            struct.pack("<I", value & 0xFFFFFFFF)
        )
    
    def alloc_dma_buffer(self, size: int) -> DMABuffer:
        """Allocate buffer (USB uses software buffer)."""
        data = np.zeros(size, dtype=np.uint8)
        return DMABuffer(
            virtual_addr=data.ctypes.data,
            physical_addr=0,
            size=size,
            data=data
        )
    
    def free_dma_buffer(self, buffer: DMABuffer) -> None:
        """Free buffer (no-op for USB, handled by Python GC)."""
        pass
    
    def dma_transfer_to_device(self, buffer: DMABuffer, dest_addr: int,
                                size: int, offset: int = 0) -> None:
        """Transfer data to device via USB bulk transfer."""
        if not self._usb_dev:
            raise DeviceError("Device not open")
        
        # Send address header
        header = struct.pack("<II", dest_addr, size)
        self._usb_dev.write(0x01, header)  # EP1 OUT
        
        # Send data in chunks
        chunk_size = 16384
        data = buffer.data[offset:offset + size]
        for i in range(0, len(data), chunk_size):
            self._usb_dev.write(0x01, bytes(data[i:i + chunk_size]))
    
    def dma_transfer_from_device(self, src_addr: int, buffer: DMABuffer,
                                  size: int, offset: int = 0) -> None:
        """Transfer data from device via USB bulk transfer."""
        if not self._usb_dev:
            raise DeviceError("Device not open")
        
        # Send read request
        header = struct.pack("<II", src_addr, size)
        self._usb_dev.write(0x01, header)
        
        # Read data in chunks
        chunk_size = 16384
        received = bytearray()
        while len(received) < size:
            data = self._usb_dev.read(0x81, min(chunk_size, size - len(received)))
            received.extend(data)
        
        buffer.data[offset:offset + size] = np.frombuffer(received[:size], dtype=np.uint8)



# =============================================================================
# Simulated Device Implementation
# =============================================================================

class SimulatedDevice(SiLensDevice):
    """
    Simulated SiLens device for development without hardware.
    
    Provides a functional simulation of the hardware that can be used
    for SDK development, testing, and demonstration purposes.
    """
    
    def __init__(self, latency_ms: float = 10.0, tokens_per_sec: float = 50.0):
        super().__init__()
        self.latency_ms = latency_ms
        self.tokens_per_sec = tokens_per_sec
        
        # Simulated registers
        self._registers: dict[int, int] = {
            Registers.CTRL: 0,
            Registers.STATUS: StatusBits.READY | StatusBits.INIT_DONE,
            Registers.VERSION: 0x00010000,  # v0.1.0
            Registers.BUILD_DATE: 0x20240101,
        }
        
        # Simulated memory
        self._memory = bytearray(BAR1_SIZE)
        self._dma_buffers: List[DMABuffer] = []
        
        # Inference state
        self._inference_thread: Optional[threading.Thread] = None
        self._stop_inference = threading.Event()
        self._token_queue: List[int] = []
    
    def __repr__(self) -> str:
        return "SimulatedDevice()"
    
    def open(self) -> None:
        """Open simulated device."""
        if self._is_open:
            return
        self._is_open = True
        self._registers[Registers.STATUS] = StatusBits.READY | StatusBits.INIT_DONE
        logger.info("Opened simulated SiLens device")
    
    def close(self) -> None:
        """Close simulated device."""
        self._stop_inference.set()
        if self._inference_thread:
            self._inference_thread.join(timeout=1.0)
        self._is_open = False
        logger.info("Closed simulated device")

    
    def read_reg(self, offset: int) -> int:
        """Read simulated register."""
        return self._registers.get(offset, 0)
    
    def write_reg(self, offset: int, value: int) -> None:
        """Write simulated register."""
        self._registers[offset] = value & 0xFFFFFFFF
        
        # Handle control register writes
        if offset == Registers.CTRL:
            if value & CtrlBits.RESET:
                self._handle_reset()
            elif value & CtrlBits.START_INFERENCE:
                self._handle_start_inference(bool(value & CtrlBits.STREAMING_MODE))
    
    def alloc_dma_buffer(self, size: int) -> DMABuffer:
        """Allocate simulated DMA buffer."""
        data = np.zeros(size, dtype=np.uint8)
        buffer = DMABuffer(
            virtual_addr=data.ctypes.data,
            physical_addr=len(self._dma_buffers) * 0x10000,
            size=size,
            data=data
        )
        self._dma_buffers.append(buffer)
        return buffer
    
    def free_dma_buffer(self, buffer: DMABuffer) -> None:
        """Free simulated DMA buffer."""
        if buffer in self._dma_buffers:
            self._dma_buffers.remove(buffer)
    
    def dma_transfer_to_device(self, buffer: DMABuffer, dest_addr: int,
                                size: int, offset: int = 0) -> None:
        """Simulate DMA transfer to device."""
        time.sleep(0.001)  # Simulate transfer time
        src_data = buffer.data[offset:offset + size]
        self._memory[dest_addr:dest_addr + size] = bytes(src_data)
    
    def dma_transfer_from_device(self, src_addr: int, buffer: DMABuffer,
                                  size: int, offset: int = 0) -> None:
        """Simulate DMA transfer from device."""
        time.sleep(0.001)  # Simulate transfer time
        buffer.data[offset:offset + size] = np.frombuffer(
            self._memory[src_addr:src_addr + size], dtype=np.uint8
        )
    
    def _handle_reset(self) -> None:
        """Handle reset command."""
        self._stop_inference.set()
        if self._inference_thread:
            self._inference_thread.join(timeout=1.0)
        self._registers[Registers.STATUS] = StatusBits.READY | StatusBits.INIT_DONE
        self._token_queue.clear()

    
    def _handle_start_inference(self, streaming: bool) -> None:
        """Handle start inference command."""
        self._stop_inference.clear()
        self._registers[Registers.STATUS] = StatusBits.BUSY
        
        def run_inference():
            # Simulate vision encoder processing
            time.sleep(self.latency_ms / 1000.0)
            self._registers[Registers.STATUS] |= StatusBits.VISION_DONE
            
            # Simulate token generation
            # Generate some sample tokens (would come from model in real implementation)
            sample_response = "A photo showing a scene with various objects and details."
            
            # Simple tokenization simulation (one token per character for demo)
            for char in sample_response:
                if self._stop_inference.is_set():
                    break
                
                token_id = ord(char)  # Simplified
                
                # Store token in output buffer
                self._token_queue.append(token_id)
                self._registers[Registers.STATUS] |= StatusBits.TOKEN_READY
                
                if self._interrupt_callback:
                    self._interrupt_callback(StatusBits.TOKEN_READY)
                
                time.sleep(1.0 / self.tokens_per_sec)
            
            # Append EOS token
            self._token_queue.append(2)
            self._registers[Registers.STATUS] = (
                StatusBits.READY | StatusBits.VISION_DONE | StatusBits.LLM_DONE
            )
        
        self._inference_thread = threading.Thread(target=run_inference, daemon=True)
        self._inference_thread.start()
    
    def get_next_token(self) -> Optional[int]:
        """Get next generated token (simulation helper)."""
        if self._token_queue:
            self._registers[Registers.STATUS] &= ~StatusBits.TOKEN_READY
            return self._token_queue.pop(0)
        return None
    
    def set_simulated_response(self, response: str) -> None:
        """Set the simulated response text (for testing)."""
        # This would be used in tests to control the output
        pass
