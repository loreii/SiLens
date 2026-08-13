# SiLens Card Firmware

Firmware for the SiLens FPGA prototype and ASIC accelerator card.

## Overview

This firmware runs on a soft-core processor (RISC-V) embedded in the FPGA or a dedicated control processor on the ASIC. It handles:

- PCIe configuration and initialization
- Host command processing
- Hardware control sequencing
- DMA management
- Status reporting

## Building

### Prerequisites

- RISC-V toolchain (riscv32-unknown-elf-gcc)
- Make

### Install Toolchain (Ubuntu/Debian)

```bash
sudo apt install gcc-riscv64-unknown-elf
# Or for 32-bit:
# Install from source or use SiFive/RISC-V Foundation releases
```

### Build

```bash
make
```

### Build with Debug Symbols

```bash
make DEBUG=1
```

### Clean

```bash
make clean
```

## Output Files

| File | Description |
|------|-------------|
| `silens_fw.elf` | ELF executable |
| `silens_fw.bin` | Raw binary |
| `silens_fw.hex` | Intel HEX format |
| `silens_fw.mem` | Verilog $readmemh format |
| `silens_fw.dis` | Disassembly |
| `silens_fw.map` | Linker map |

## Memory Map

| Address | Size | Description |
|---------|------|-------------|
| 0x00000000 | 16KB | Boot ROM |
| 0x00001000 | 4KB | Host-visible registers (BAR0) |
| 0x00010000 | 64KB | Main RAM |
| 0x0001F000 | 4KB | Stack |
| 0x10000000 | - | PCIe controller |
| 0x20000000 | - | Vision encoder |
| 0x30000000 | - | Language model |
| 0x40000000 | - | DMA engine |

## Host Register Interface

The firmware exposes registers to the host via PCIe BAR0:

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| 0x000 | CTRL | R/W | Control register |
| 0x004 | STATUS | R | Status register |
| 0x008 | IMG_ADDR | R/W | Image buffer address |
| 0x00C | IMG_SIZE | R/W | Image dimensions |
| 0x010 | OUT_ADDR | R/W | Output buffer address |
| 0x014 | OUT_LEN | R | Output token count |
| 0x018 | TOKEN_OUT | R | Current output token |
| 0x01C | TOKEN_VALID | R | Token valid flag |
| 0x028 | INT_STATUS | R/W | Interrupt status |
| 0x02C | INT_ENABLE | R/W | Interrupt enable |
| 0x100 | DMA_CTRL | R/W | DMA control |
| 0x104 | DMA_STATUS | R | DMA status |
| 0x1F0 | VERSION | R | Firmware version |

## State Machine

```
                    ┌──────────┐
                    │   IDLE   │◄───────────────────┐
                    └────┬─────┘                    │
                         │ START                   │
                         ▼                         │
                    ┌──────────┐                   │
                    │  VISION  │                   │
                    └────┬─────┘                   │
                         │ DONE                    │
                         ▼                         │
                    ┌──────────┐                   │
                    │ PROJECT  │                   │
                    └────┬─────┘                   │
                         │                         │
                         ▼                         │
                    ┌──────────┐                   │
               ┌───►│   LLM    │───┐               │
               │    └──────────┘   │               │
               │         │         │               │
               │     token     EOS/MAX             │
               │         │         │               │
               └─────────┘         ▼               │
                              ┌──────────┐         │
                              │  OUTPUT  │─────────┘
                              └──────────┘
```

## Adapting for Different Targets

### Different Processor Core

1. Modify `ARCH_FLAGS` in Makefile
2. Update startup.S for your ISA
3. Adjust linker.ld memory regions

### Different Memory Layout

1. Edit linker.ld with your memory addresses
2. Update base addresses in main.c

### Adding New Peripherals

1. Add register definitions to main.c
2. Add initialization in hardware_init()
3. Add command handling if needed

## License

Apache-2.0
