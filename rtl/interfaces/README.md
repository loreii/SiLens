# SiLens Interface Modules

This directory contains the external interface controllers for the SiLens SoC.

## Modules

### silens_host_interface.v

Parallel host interface controller for FPGA bridge communication.

**Features:**
- 32-bit parallel data bus @ 100 MHz (400 MB/s peak)
- 16-bit register address space
- Token input/output FIFOs with clock domain crossing
- Pixel streaming FIFO for image input
- Interrupt generation
- Control signal synchronization

**Register Map:**

| Address | Name | Description |
|---------|------|-------------|
| 0x0000 | CONTROL | Frame/seq/gen start, abort |
| 0x0004 | STATUS | State, busy, error flags |
| 0x0008 | TOKEN_WR | Input token FIFO |
| 0x000C | TOKEN_RD | Output token FIFO |
| 0x0010 | DMA_BASE | Image DMA base |
| 0x001C | IRQ_ENABLE | Interrupt enable |
| 0x0020 | IRQ_STATUS | Interrupt status (W1C) |
| 0x0024 | VERSION | Hardware version |
| 0x0028 | PIXEL_WR | Pixel input FIFO |
| 0x002C | FIFO_STATUS | FIFO flags |

### silens_spi_slave.v

SPI slave for configuration and debug access.

**Features:**
- SPI Mode 0 (CPOL=0, CPHA=0)
- MSB first, max 10 MHz
- 8-bit register read/write
- Basic status monitoring

**Protocol:**
- Write: `[0][7-bit addr][8-bit data]` = 16 bits
- Read: `[1][7-bit addr]` → `[8-bit data]`

## Integration

These modules are instantiated in `silens_soc.v` and connect to:
- FPGA bridge (parallel interface) for PCIe/USB connectivity
- External SPI master for configuration
- Internal VLM core for token/pixel streaming

## Clock Domain Crossing

The host interface implements proper CDC using:
- Gray-coded async FIFOs for token streams
- Multi-stage synchronizers for control signals
- Double-FF synchronizers for status signals

All FIFOs are 256 entries deep by default.
