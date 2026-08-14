# SiLens RTL Design

This directory contains the complete RTL design for the SiLens 800mm² Vision-Language AI Accelerator targeting SkyWater SKY130.

## Directory Structure

```
rtl/
├── top/                    # Top-level SoC
│   ├── silens_soc.v        # 800mm² full-custom SoC
│   └── silens_top.v        # Original FPGA-targeted top
├── core/                   # Processing core
│   └── silens_vlm_core.v   # VLM pipeline coordinator
├── vision/                 # Vision encoder
│   └── vision_encoder_top.v # SigLIP-B/16 encoder
├── llm/                    # Language model
│   ├── llm_decoder_top.v   # SmolLM2-135M decoder
│   └── ...                 # Transformer blocks
├── projector/              # Multimodal projector
│   └── projector.v         # 768→576 linear projection
├── memory/                 # Memory interfaces
│   └── ddr3_controller.v   # DDR3-1066 controller
├── interfaces/             # External interfaces
│   ├── silens_host_interface.v  # Parallel host bus
│   └── silens_spi_slave.v  # SPI configuration
├── common/                 # Shared modules
│   ├── clock_gen.v         # Clock generation
│   ├── reset_sync.v        # Reset synchronizer
│   ├── ternary_mac.v       # Ternary MAC unit
│   ├── rms_norm.v          # RMS normalization
│   ├── layer_norm.v        # Layer normalization
│   └── ...                 # Other utilities
├── attention/              # Attention mechanisms
├── skywater/               # Caravel-specific (shuttle)
└── fpga/                   # FPGA wrappers
```

## Design Hierarchy

```
silens_soc (top)
├── silens_clock_gen
├── silens_reset_sync (×3)
├── silens_ddr3_controller
├── silens_host_interface
├── silens_spi_slave
└── silens_vlm_core
    ├── vision_encoder_top (SigLIP-B/16, 93M params)
    ├── projector (18M params)
    └── llm_decoder_top (SmolLM2-135M, 135M params)
```

## Key Specifications

| Parameter | Value |
|-----------|-------|
| Process | SkyWater SKY130 (130nm) |
| Die Size | 800mm² |
| Clock | 100 MHz |
| Power | 25W TDP |
| Model | SmolVLM-256M (246M params) |
| Quantization | Ternary (-1, 0, +1) |
| Weights | 61.5MB hardwired |

## Building

### Simulation (Icarus Verilog)
```bash
make sim
```

### FPGA Synthesis (Xilinx)
```bash
make fpga
```

### ASIC Synthesis (OpenLane)
```bash
cd ../openlane/silens_soc
flow.tcl -design .
```

## Testing

Run the complete test suite:
```bash
make test
```

Run individual tests:
```bash
make test_vision
make test_llm
make test_projector
```

## Documentation

- [SoC Architecture](../docs/architecture/SILENS_SOC_ARCHITECTURE.md)
- [SkyWater Capabilities](../docs/SKYWATER_CAPABILITIES_CONFIRMED.md)
- [Development Plan](../PLAN.md)

## License

Apache 2.0
