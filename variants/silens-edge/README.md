# SiLens Edge - Ultra-Fast Vision Classifier

> 50mm² hardwired AI accelerator for industrial edge deployment

## Overview

SiLens Edge is a compact vision classifier optimized for embedded and industrial applications. It runs inference in under 1ms, classifying images at 1000 FPS with just 500mW active power.

## Specifications

| Parameter | Value |
|-----------|-------|
| Die Size | 50mm² (7mm × 7mm) |
| Process | SkyWater SKY130 130nm |
| Clock | 200MHz |
| Power | 3W TDP, <500mW active |
| Package | QFN-48 |
| **Model** | TinyVLM-20M |
| Vision Encoder | NanoViT-12M (6 layers, 192-dim) |
| Classifier | 7M params (4 layers, 128-dim) |

## Performance

| Metric | Target |
|--------|--------|
| Latency | <1ms per image |
| Throughput | 1000 FPS |
| Image Size | 224×224 RGB |
| Classes | 1000 (ImageNet-style) |

## Block Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SiLens Edge SoC (50mm²)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    7000µm × 7000µm                        │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                │   │
│  │  │   Vision Nano   │  │ Classifier Head │                │   │
│  │  │    (~15mm²)     │  │    (~10mm²)     │                │   │
│  │  │  NanoViT-12M    │──│   4-layer MLP   │                │   │
│  │  │  6 transformer  │  │   128-dim       │                │   │
│  │  │  192-dim        │  │   1000 classes  │                │   │
│  │  └─────────────────┘  └─────────────────┘                │   │
│  │                                                          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                │   │
│  │  │   IO Edge       │  │  SRAM 256KB     │                │   │
│  │  │    (~5mm²)      │  │   (~10mm²)      │                │   │
│  │  │  SPI Slave      │  │  Dual-port      │                │   │
│  │  │  I2C Config     │  │  Activation     │                │   │
│  │  │  8 GPIO         │  │  Buffer         │                │   │
│  │  └─────────────────┘  └─────────────────┘                │   │
│  │                                                          │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │            PLL + Reset + Clocks (~10mm²)            │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Interfaces

### SPI Slave (Primary)
- 50MHz max clock
- Mode 0 (CPOL=0, CPHA=0)
- Commands: Write registers, Read registers, Write image burst, Read result

### I2C Slave (Configuration)
- Address: 0x50 (configurable)
- 100/400 kHz support
- Register map for configuration

### GPIO (8 pins)
| Pin | Function |
|-----|----------|
| 0 | TRIG_IN - External trigger input |
| 1-4 | CLASS[3:0] - Classification result |
| 5 | BUSY - Inference in progress |
| 6 | ERROR - Error indicator |
| 7 | IRQ_N - Interrupt (active low) |

## Use Cases

- Industrial defect detection
- Object presence detection (yes/no)
- Safety triggers (person detected)
- Quality control classification
- Drone obstacle detection
- Robotics scene classification

## Building

```bash
# Build shared Level 1-2 primitives (required once)
cd openlane
make level1 level2

# Build Edge-specific Level 3-4
make VARIANT=silens-edge level3
make VARIANT=silens-edge level4

# Or build everything
make VARIANT=silens-edge all
```

## Directory Structure

```
variants/silens-edge/
├── config.json             # Variant configuration
├── README.md               # This file
├── docs/                   # Edge-specific documentation
├── openlane/
│   ├── level3/
│   │   ├── vision_nano/    # NanoViT encoder (~15mm²)
│   │   ├── classifier_head/ # MLP classifier (~10mm²)
│   │   ├── io_edge/        # SPI/I2C/GPIO (~5mm²)
│   │   └── sram_256kb/     # Activation SRAM (~10mm²)
│   └── level4/
│       └── silens_edge_soc/ # Top integration (50mm²)
└── rtl/                    # Variant-specific RTL (if any)
```

## Comparison with SiLens VLM

| Feature | SiLens Edge | SiLens VLM |
|---------|-------------|------------|
| Die Size | 50mm² | 800mm² |
| Model | TinyVLM-20M | PaliGemma-3B |
| Output | Single class (1000) | Text generation |
| Memory | On-chip only | DDR3 external |
| Interface | SPI/I2C/GPIO | PCIe/Parallel |
| Package | QFN-48 | BGA-900 |
| Power | 3W TDP | 25W TDP |
| Use Case | Embedded classification | Full VLM inference |

## License

Apache 2.0 - See [LICENSE](../../LICENSE)
