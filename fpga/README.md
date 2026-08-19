# SiLens FPGA Support

This directory contains FPGA synthesis support files for prototyping both SiLens accelerator variants:

- **SiLens-VLM**: Large 800mm² ASIC target for high-performance multimodal LLM inference with PCIe interface
- **SiLens-Edge**: Compact 50mm² ASIC target for embedded vision applications with SPI/I2C interface

## Overview

| Variant | ASIC Target | Interface | Use Case | FPGA Requirements |
|---------|-------------|-----------|----------|-------------------|
| **SiLens-VLM** | 800mm² | PCIe Gen3 x4/x8 | Server/desktop multimodal LLM | Large FPGAs (Kintex-7, Arria 10) |
| **SiLens-Edge** | 50mm² | SPI + I2C | Embedded vision, MCU companion | Small FPGAs (Artix-7 35T, iCE40) |

## Directory Structure

```
fpga/
├── README.md
├── xilinx/                          # SiLens-VLM Xilinx files
│   ├── silens_artix7.xdc            # Constraints for Artix-7 200T
│   ├── silens_kintex7.xdc           # Constraints for Kintex-7 325T
│   ├── silens_fpga_wrapper.v        # Top-level FPGA wrapper with MMCM
│   └── synth_vivado.tcl             # Vivado synthesis script
├── intel/                           # SiLens-VLM Intel files
│   ├── silens_cyclone10.sdc         # Constraints for Cyclone 10 GX
│   ├── silens_arria10.sdc           # Constraints for Arria 10 GX
│   └── silens_fpga_wrapper_intel.v  # Intel FPGA wrapper
└── edge/                            # SiLens-Edge files (NEW)
    ├── silens_edge_fpga_wrapper.v   # Edge-specific wrapper (SPI/I2C)
    └── silens_edge_artix7_35t.xdc   # Constraints for Arty A7-35T
```

---

## SiLens-VLM FPGA Support

The VLM variant targets high-performance multimodal inference with PCIe connectivity. Due to the large model size (256M parameters), FPGA prototypes demonstrate the architecture with reduced model size or layer count.

### Supported Development Boards

#### Xilinx

| Board | Device | Logic Cells | DSP | BRAM | PCIe |
|-------|--------|-------------|-----|------|------|
| Nexys Video / Arty A7-200T | Artix-7 200T | ~215K | 740 | 13.14 Mb | Gen2 x4 |
| KC705 | Kintex-7 325T | ~325K | 840 | 16.02 Mb | Gen2 x8 / Gen3 x4 |

#### Intel

| Board | Device | Logic Elements | DSP | Memory | PCIe |
|-------|--------|----------------|-----|--------|------|
| Cyclone 10 GX Dev Kit | Cyclone 10 GX | ~220K | 192 | 11 Mb | Gen3 x4 |
| Arria 10 GX Dev Kit | Arria 10 GX | ~660K | 1,687 | 42 Mb | Gen3 x8 |

### VLM Resource Estimates

For a scaled-down prototype (reduced sequence length, single transformer block):

| Resource | Artix-7 200T | Kintex-7 325T | Estimated Usage |
|----------|--------------|---------------|-----------------|
| LUTs     | 134,600      | 203,800       | ~60-80%         |
| FFs      | 269,200      | 407,600       | ~40-60%         |
| BRAM     | 365 (36Kb)   | 445 (36Kb)    | ~80-95%         |
| DSP      | 740          | 840           | ~30-50%         |

### Building VLM for Xilinx

#### Prerequisites
- Vivado 2022.2 or later
- License for target device

#### Steps

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

### Building VLM for Intel

#### Prerequisites
- Quartus Prime 22.1 or later
- License for target device

#### Steps

1. Create a new Quartus project
2. Add RTL sources from `rtl/` directory
3. Add `silens_fpga_wrapper_intel.v` as top-level
4. Add SDC constraints from `intel/`
5. Run compilation

### VLM Clock Architecture

#### Xilinx
- **Input**: 200 MHz differential LVDS
- **Core clock**: 100 MHz (from MMCM)
- **PCIe clock**: 250 MHz (from MMCM)

#### Intel
- **Input**: 50 MHz or 100 MHz oscillator
- **Core clock**: 100 MHz (from IOPLL)
- **Fast clock**: 200 MHz (from IOPLL)

### PCIe Integration

The FPGA wrappers include placeholder connections for vendor PCIe hard IP:

#### Xilinx
- Use PCIe 7-Series or UltraScale+ IP
- Configure as Gen3 x4 endpoint
- Enable 128-bit AXI-Stream interface

#### Intel
- Use Hard IP for PCI Express
- Configure as Gen3 x4 endpoint
- Enable Avalon-ST interface

### VLM Memory Considerations

FPGA BRAMs are used for:
- Activation buffers (double-buffered)
- KV cache (reduced context length)
- Embedding cache (subset of vocabulary)

For full model support, connect external DDR3/DDR4 memory using MIG (Xilinx) or EMIF (Intel).

---

## SiLens-Edge FPGA Support

The Edge variant is optimized for embedded applications with a much smaller footprint. It uses simple SPI/I2C interfaces for integration with microcontrollers and supports open-source FPGA toolchains.

### Supported Development Boards

| Board | Device | LUTs | FFs | BRAM | DSP | Cost |
|-------|--------|------|-----|------|-----|------|
| **Arty A7-35T** | Artix-7 35T | 33,280 | 41,600 | 50×36Kb | 90 | ~$130 |
| Basys 3 | Artix-7 35T | 33,280 | 41,600 | 50×36Kb | 90 | ~$150 |
| **UPduino v3.1** | iCE40 UP5K | 5,280 | 5,280 | 30×4Kb | 8 | ~$20 |
| TinyFPGA BX | iCE40 LP8K | 7,680 | 7,680 | 32×4Kb | 0 | ~$40 |
| OrangeCrab | ECP5-25F | 24,000 | 24,000 | 56×18Kb | 28 | ~$80 |

### Edge Resource Estimates

For SiLens-Edge (NanoViT-12M + Classifier) on small FPGAs:

| Resource | Artix-7 35T Available | Edge Usage | Utilization |
|----------|----------------------|------------|-------------|
| LUTs     | 33,280               | ~20,000    | 60%         |
| FFs      | 41,600               | ~15,000    | 45%         |
| BRAM     | 50 (36Kb)            | ~40        | 80%         |
| DSP      | 90                   | ~30        | 33%         |

| Resource | iCE40 UP5K Available | Edge Usage (Minimal) | Utilization |
|----------|---------------------|---------------------|-------------|
| LUTs     | 5,280               | ~4,500              | 85%         |
| FFs      | 5,280               | ~4,000              | 76%         |
| BRAM     | 30 (4Kb)            | ~28                 | 93%         |
| DSP      | 8                   | ~8                  | 100%        |

> **Note**: iCE40 requires aggressive optimizations (reduced precision, sequential processing) to fit. Artix-7 35T is recommended for full-featured Edge prototyping.

### Building Edge for Xilinx (Vivado)

#### Prerequisites
- Vivado 2022.2 or later (WebPACK edition is sufficient for Artix-7 35T)
- Digilent board files installed

#### Steps

1. Create a new Vivado project targeting `xc7a35tcsg324-1`:
```bash
vivado -mode tcl
```

2. In Tcl console:
```tcl
create_project silens_edge ./silens_edge -part xc7a35tcsg324-1
add_files -norecurse {
    ../rtl/edge/silens_edge_soc.v
    ../rtl/edge/silens_edge_core.v
    ../rtl/common/spi_slave.v
    ../rtl/common/i2c_slave.v
    fpga/edge/silens_edge_fpga_wrapper.v
}
add_files -fileset constrs_1 fpga/edge/silens_edge_artix7_35t.xdc
set_property top silens_edge_fpga_wrapper [current_fileset]

# Enable XILINX define for platform-specific code
set_property verilog_define {XILINX=1} [current_fileset]

# Run synthesis with area optimization
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Run implementation
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
```

3. Program the board:
```tcl
open_hw_manager
connect_hw_server
open_hw_target
set_property PROGRAM.FILE {./silens_edge/silens_edge.runs/impl_1/silens_edge_fpga_wrapper.bit} [current_hw_device]
program_hw_devices [current_hw_device]
```

### Building Edge with Open-Source Tools (iCE40)

For Lattice iCE40 FPGAs, use the fully open-source Yosys/nextpnr toolchain.

#### Prerequisites
```bash
# Ubuntu/Debian
sudo apt install yosys nextpnr-ice40 icestorm

# macOS (Homebrew)
brew install yosys nextpnr icestorm

# Or build from source: https://github.com/YosysHQ/oss-cad-suite-build
```

#### Build Steps

1. Synthesize with Yosys:
```bash
cd fpga/edge

yosys -p "
    read_verilog -DICE40 silens_edge_fpga_wrapper.v
    read_verilog ../../rtl/edge/silens_edge_soc.v
    read_verilog ../../rtl/edge/silens_edge_core.v
    read_verilog ../../rtl/common/spi_slave.v
    read_verilog ../../rtl/common/i2c_slave.v
    synth_ice40 -top silens_edge_fpga_wrapper -json silens_edge_ice40.json
"
```

2. Place and route with nextpnr:
```bash
nextpnr-ice40 --up5k --package sg48 \
    --json silens_edge_ice40.json \
    --pcf silens_edge_ice40_up5k.pcf \
    --asc silens_edge_ice40.asc
```

3. Generate bitstream:
```bash
icepack silens_edge_ice40.asc silens_edge_ice40.bin
```

4. Program the UPduino:
```bash
iceprog silens_edge_ice40.bin
```

> **Note**: A PCF constraints file for iCE40 (`silens_edge_ice40_up5k.pcf`) will be added in a future update.

### Edge Quick Start Demo

Once the bitstream is programmed on an Arty A7-35T:

1. **Connect an MCU** (e.g., STM32, ESP32, Arduino) to the Pmod JA header:
   - JA1 → SPI CS_N
   - JA2 → SPI MOSI
   - JA3 → SPI MISO
   - JA4 → SPI SCLK

2. **Observe status LEDs**:
   - LED0: PLL locked (system ready)
   - LED1: Inference busy
   - LED2: Classification valid
   - LED3: Error indicator

3. **Trigger inference**:
   - Send image data over SPI
   - Or pulse GPIO[0] (Pmod JC pin 1) to trigger with test pattern

4. **Read classification result**:
   - Poll `class_valid` signal
   - Read `class_id[9:0]` from Pmod JD header or via SPI

### Edge Clock Architecture

- **Input**: 100 MHz single-ended LVCMOS (board oscillator)
- **Core clock**: 100 MHz (from MMCM)
- **SPI clock**: Up to 50 MHz (external, asynchronous)
- **I2C clock**: Up to 400 kHz (standard/fast mode)

### Edge Interface Pinout (Arty A7-35T)

| Interface | Pmod | Pins | Description |
|-----------|------|------|-------------|
| SPI | JA | JA1-JA4 | CS_N, MOSI, MISO, SCLK |
| I2C | JB | JB1-JB2 | SCL, SDA (with pull-ups) |
| GPIO | JC | JC1-JC8 | 8-bit bidirectional |
| Class Output | JD | JD1-JD8 | class_id[7:0], valid, busy |

---

## Interface Comparison

| Feature | SiLens-VLM (PCIe) | SiLens-Edge (SPI/I2C) |
|---------|-------------------|------------------------|
| **Bandwidth** | 8-16 GB/s (Gen3 x4/x8) | 6.25 MB/s (SPI @ 50MHz) |
| **Latency** | Low (~1µs) | Moderate (~20µs) |
| **Host Requirements** | x86/ARM64 with PCIe slot | Any MCU with SPI |
| **Power** | 15-75W (FPGA dependent) | 0.5-2W |
| **Configuration** | PCIe BAR registers | I2C slave registers |
| **Data Transfer** | DMA via PCIe | SPI burst transactions |
| **Typical Use Case** | Server inference, desktop AI | IoT, robotics, wearables |
| **Supported FPGAs** | Kintex-7, Arria 10 | Artix-7 35T, iCE40 UP5K |

---

## Debug Features

Both variants support:
- ILA/SignalTap integration points
- Status LEDs for state machine visibility
- UART debug output (directly accessible)

### VLM-Specific
- Debug register access via PCIe BAR0

### Edge-Specific
- GPIO-based trigger and status
- SPI command interface for register reads
- Classification output directly on Pmod header

---

## Simulation

Before synthesis, simulate using cocotb:

```bash
cd rtl/tb
make test_all    # Run all tests
make test_edge   # Run Edge-specific tests
```

Or with Vivado simulator:
```bash
vivado -mode tcl
create_project sim_proj ./sim_proj -part xc7a35tcsg324-1
add_files -norecurse ../rtl/edge/
set_property top silens_edge_tb [get_filesets sim_1]
launch_simulation
```

---

## Known Limitations

### General
1. **Model Size**: Full models won't fit in FPGA BRAMs. Use:
   - Reduced layer count (e.g., 4 layers instead of 30)
   - Smaller dimensions
   - External memory for weights (VLM only)

2. **Performance**: FPGA clock speeds (~100-200 MHz) are lower than ASIC target. Expect:
   - ~10-20x slower inference than ASIC target
   - Useful for functional verification, not performance benchmarking

### VLM-Specific
3. **PCIe**: Vendor-specific hard IP required. Placeholder modules provided.
4. **Memory Bandwidth**: External DDR required for full model; MIG/EMIF integration needed.

### Edge-Specific
5. **iCE40 Constraints**: Very tight resource fit; requires reduced precision mode.
6. **SPI Throughput**: 50 MHz SPI limits image transfer to ~6 frames/second for 224×224 images.
7. **No External Memory**: Edge variant is designed for on-chip weights only.

---

## License

Apache 2.0 - See LICENSE file in repository root.
