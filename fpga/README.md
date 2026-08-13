# SiLens FPGA Support

This directory contains FPGA synthesis support files for prototyping the SiLens accelerator on Xilinx and Intel FPGAs.

## Directory Structure

```
fpga/
├── xilinx/
│   ├── silens_artix7.xdc      # Constraints for Artix-7 (Nexys Video)
│   ├── silens_kintex7.xdc     # Constraints for Kintex-7 (KC705)
│   ├── silens_fpga_wrapper.v  # Top-level FPGA wrapper with MMCM
│   └── synth_vivado.tcl       # Vivado synthesis script
├── intel/
│   ├── silens_cyclone10.sdc   # Constraints for Cyclone 10 GX
│   ├── silens_arria10.sdc     # Constraints for Arria 10 GX
│   └── silens_fpga_wrapper_intel.v  # Intel FPGA wrapper
└── README.md
```

## Supported Development Boards

### Xilinx
- **Artix-7 200T** (Nexys Video, Arty A7-200T)
  - ~215K logic cells
  - 740 DSP slices
  - 13.14 Mb block RAM
  - PCIe Gen2 x4 or Gen3 x4

- **Kintex-7 325T** (KC705)
  - ~325K logic cells
  - 840 DSP slices
  - 16.02 Mb block RAM
  - PCIe Gen2 x8 or Gen3 x4

### Intel
- **Cyclone 10 GX** (Development Kit)
  - ~220K logic elements
  - 192 variable-precision DSP blocks
  - 11 Mb embedded memory
  - PCIe Gen3 x4

- **Arria 10 GX** (Development Kit)
  - ~660K logic elements
  - 1,687 variable-precision DSP blocks
  - 42 Mb embedded memory
  - PCIe Gen3 x8

## FPGA Resource Estimates

For a scaled-down prototype (reduced sequence length, single transformer block):

| Resource | Artix-7 200T | Kintex-7 325T | Estimate Usage |
|----------|--------------|---------------|----------------|
| LUTs     | 134,600      | 203,800       | ~60-80%        |
| FFs      | 269,200      | 407,600       | ~40-60%        |
| BRAM     | 365 (36Kb)   | 445 (36Kb)    | ~80-95%        |
| DSP      | 740          | 840           | ~30-50%        |

**Note**: Full SiLens requires ASIC implementation. FPGA prototypes demonstrate the architecture with reduced model size or layer count.

## Building for Xilinx

### Prerequisites
- Vivado 2022.2 or later
- License for target device

### Steps

1. Open Vivado and create a new project:
```tcl
source fpga/xilinx/synth_vivado.tcl
```

2. Or manually:
```bash
cd fpga/xilinx
vivado -mode tcl
source synth_vivado.tcl
```

3. Run synthesis, implementation, and generate bitstream.

## Building for Intel

### Prerequisites
- Quartus Prime 22.1 or later
- License for target device

### Steps

1. Create a new Quartus project
2. Add RTL sources from `rtl/` directory
3. Add `silens_fpga_wrapper_intel.v` as top-level
4. Add SDC constraints from `intel/`
5. Run compilation

## Clock Architecture

### Xilinx
- **Input**: 200 MHz differential LVDS
- **Core clock**: 100 MHz (from MMCM)
- **PCIe clock**: 250 MHz (from MMCM)

### Intel
- **Input**: 50 MHz or 100 MHz oscillator
- **Core clock**: 100 MHz (from IOPLL)
- **Fast clock**: 200 MHz (from IOPLL)

## PCIe Integration

The FPGA wrappers include placeholder connections for vendor PCIe hard IP:

### Xilinx
- Use PCIe 7-Series or UltraScale+ IP
- Configure as Gen3 x4 endpoint
- Enable 128-bit AXI-Stream interface

### Intel
- Use Hard IP for PCI Express
- Configure as Gen3 x4 endpoint
- Enable Avalon-ST interface

## Memory Considerations

FPGA BRAMs are used for:
- Activation buffers (double-buffered)
- KV cache (reduced context length)
- Embedding cache (subset of vocabulary)

For full model support, connect external DDR3/DDR4 memory using MIG (Xilinx) or EMIF (Intel).

## Debug Features

- ILA/SignalTap integration points
- Debug register access via PCIe BAR0
- Status LEDs for state machine visibility
- UART debug output (optional)

## Simulation

Before synthesis, simulate using:

```bash
cd rtl/tb
make test_all    # Run cocotb tests
```

Or with Vivado simulator:
```bash
vivado -mode tcl
create_project sim_proj ./sim_proj -part xc7a200tfbg676-1
add_files -norecurse ../rtl/
set_property top silens_top_tb [get_filesets sim_1]
launch_simulation
```

## Known Limitations

1. **Model Size**: Full 256M parameter model won't fit in FPGA BRAMs. Use:
   - Reduced layer count (e.g., 4 layers instead of 30)
   - Smaller dimensions (e.g., 256 instead of 576)
   - External memory for weights

2. **Performance**: FPGA clock speeds (~100-200 MHz) are lower than ASIC target. Expect:
   - ~10-20x slower inference than ASIC target
   - Useful for functional verification, not performance benchmarking

3. **PCIe**: Vendor-specific hard IP required. Placeholder modules provided.

## License

Apache 2.0 - See LICENSE file in repository root.
