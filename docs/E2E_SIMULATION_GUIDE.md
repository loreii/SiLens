# SiLens End-to-End RTL Simulation Guide

This guide explains how to run the complete end-to-end simulation pipeline for the SiLens vision-language AI accelerator. The pipeline verifies that ternary-quantized weights can be loaded into Verilog RTL and simulated correctly.

## Overview

The E2E pipeline tests:
1. **Ternary Quantization** - Convert model weights to {-1, 0, +1}
2. **RTL Compilation** - Compile Verilog modules with Icarus Verilog
3. **Verilog Simulation** - Run hardware simulation
4. **Python/cocotb Integration** - Verify with Python-based testbenches

## Prerequisites

### Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Test framework |
| Icarus Verilog | 13.0+ | RTL simulation |
| cocotb | 2.0+ | Python-Verilog testing |
| NumPy | 1.26+ | Numerical operations |
| Make | Any | Build automation |

### Installation

#### macOS (Homebrew)

```bash
# Install Icarus Verilog
brew install icarus-verilog

# Install Python dependencies
pip install cocotb numpy
```

#### Ubuntu/Debian

```bash
# Install Icarus Verilog
sudo apt-get update
sudo apt-get install iverilog

# Install Python dependencies
pip install cocotb numpy
```

#### Windows (with WSL2)

```bash
# In WSL2 Ubuntu terminal
sudo apt-get update
sudo apt-get install iverilog
pip install cocotb numpy
```

### Verify Installation

```bash
# Check Icarus Verilog
iverilog -V
# Expected: Icarus Verilog version 13.0 (stable) (v13_0)

# Check cocotb
python -c "import cocotb; print(cocotb.__version__)"
# Expected: 2.0.1 or higher

# Check numpy
python -c "import numpy; print(numpy.__version__)"
# Expected: 1.26.x or higher
```

## Running the E2E Pipeline Test

### Quick Start

```bash
cd SiLens
python test_e2e_pipeline.py
```

### Expected Output

```
======================================================================
  SiLens End-to-End Pipeline Test
======================================================================

This test verifies the complete pipeline:
  1. Check simulation environment
  2. Quantize weights to ternary (1-bit)
  3. Compile RTL with Icarus Verilog
  4. Run simple Verilog simulation
  5. Run cocotb Python tests

[Step 1] Checking simulation environment
  ✓ Icarus Verilog: Icarus Verilog version 13.0 (stable) (v13_0)
  ✓ cocotb: 2.0.1
  ✓ numpy: 1.26.4
  ✓ make: available

[Step 2] Quantizing test weights to ternary
  ℹ vision.patch_embed: (64, 48) -> sparsity=42.4%
  ℹ vision.attn.qkv: (192, 64) -> sparsity=42.4%
  ... (13 layers total)
  ✓ Quantized 13 layers, 97,280 params
  ✓ Overall sparsity: 42.3%

[Step 3] Compiling RTL with Icarus Verilog
  ℹ Found 10 core Verilog files
  ✓ Compiled to: /tmp/silens_e2e_test_xxx/build/silens_sim

[Step 4] Running Verilog simulation
  ✓ Simulation completed successfully!
  ✓ Waveform saved to: /tmp/silens_e2e_test_xxx/build/simple_test.vcd

[Step 5] Running cocotb-based Python test
  ✓ cocotb test passed!
  ** TESTS=7 PASS=7 FAIL=0 SKIP=0

======================================================================
  Test Summary
======================================================================

  environment          [PASS]
  quantization         [PASS]
  compile              [PASS]
  simulation           [PASS]
  cocotb               [PASS]

All tests passed! ✓
```

## What Gets Tested

### 1. Quantization (Step 2)

Converts floating-point weights to ternary format:
- **Encoding**: `-1` → `0b10`, `0` → `0b00`, `+1` → `0b01`
- **Sparsity**: ~42% of weights become zero (model compression)
- **Output**: `.hex` files for Verilog `$readmemh`

Test layers:
| Layer | Shape | Purpose |
|-------|-------|---------|
| vision.patch_embed | (64, 48) | Image patch projection |
| vision.attn.qkv | (192, 64) | Attention Q/K/V projection |
| vision.attn.proj | (64, 64) | Attention output projection |
| vision.mlp.fc1 | (256, 64) | Vision MLP up-projection |
| vision.mlp.fc2 | (64, 256) | Vision MLP down-projection |
| projector.linear | (64, 64) | Vision-to-LLM projection |
| llm.attn.* | (64, 64) | LLM attention weights |
| llm.mlp.* | (128, 64) | LLM feed-forward weights |

### 2. RTL Compilation (Step 3)

Compiles these core modules:
- `popcount.v` - Bit counting for binary operations
- `ternary_mac.v` - Multiply-accumulate for ternary weights
- `binary_dot_product.v` - XNOR-popcount dot product
- `layer_norm.v` - Layer normalization
- `softmax_approx.v` - Piecewise-linear softmax
- `gelu_approx.v` - GELU activation approximation
- `rms_norm.v` - RMS normalization (LLaMA-style)
- `simd_vector_unit.v` - Vector operations
- `axi_interface.v` - Memory interface
- `power_controller.v` - Power management

### 3. Verilog Simulation (Step 4)

Tests fundamental operations:

**Popcount Test:**
```
Input: 0x0000000000000001, Count: 1 (expected: 1)
Input: 0xFFFFFFFFFFFFFFFF, Count: 64 (expected: 64)
Input: 0xAAAAAAAAAAAAAAAA, Count: 32 (expected: 32)
```

**Ternary MAC Test:**
```
All +1 weights, activations=1: result = 16
All -1 weights, activations=1: result = -16
All zero weights: result = 0
```

### 4. Cocotb Python Tests (Step 5)

7 comprehensive tests for the popcount module:

| Test | Description | Result |
|------|-------------|--------|
| test_popcount_all_zeros | All bits = 0 | count = 0 |
| test_popcount_all_ones | All bits = 1 | count = 512 |
| test_popcount_single_bit | One bit at each position | count = 1 |
| test_popcount_alternating | 0b0101... pattern | count = 256 |
| test_popcount_random | 100 random values | Matches Python |
| test_popcount_power_of_two | 2^n - 1 patterns | count = n |
| test_popcount_edge_cases | Neural network patterns | Correct |

## Running Individual Tests

### Cocotb Tests Only

```bash
cd SiLens/rtl/tb
make test-popcount      # Test popcount module
make test-ternary-mac   # Test ternary MAC
make test-softmax       # Test softmax approximation
make test-gelu          # Test GELU activation
make test-layernorm     # Test layer normalization
```

### View Waveforms

After running simulation:
```bash
# Install GTKWave if needed
brew install gtkwave  # macOS
sudo apt install gtkwave  # Ubuntu

# Open waveform
gtkwave /tmp/silens_e2e_test_xxx/build/simple_test.vcd
```

## Troubleshooting

### Common Issues

#### "iverilog not found"
```bash
# macOS
brew install icarus-verilog

# Ubuntu
sudo apt-get install iverilog
```

#### "cocotb not found"
```bash
pip install cocotb
```

#### "Compilation failed: generate is a reserved word"
The RTL files have been fixed. If you see this, pull the latest code:
```bash
git pull origin main
```

#### Cocotb test failures with "in_"
The test files have been updated to use `getattr(dut, 'in')` for signal access.

### Performance Notes

- Full test takes ~10-30 seconds
- Cocotb tests run 7 tests × ~200ns simulated time
- Waveform files can be 1-10 MB

## Architecture Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                    test_e2e_pipeline.py                         │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: check_environment()                                    │
│    └── Verifies: iverilog, cocotb, numpy, make                  │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: quantize_test_weights()                                │
│    ├── TernaryQuantizer (model/conversion/quantize_ternary.py)  │
│    └── export_ternary_to_hex() → .hex files                     │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: compile_rtl()                                          │
│    └── iverilog -g2012 rtl/common/*.v                           │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: run_simple_simulation()                                │
│    ├── Generate testbench (simple_test.v)                       │
│    ├── Compile: iverilog -o simple_test_sim                     │
│    └── Run: vvp simple_test_sim                                 │
├─────────────────────────────────────────────────────────────────┤
│  Step 5: run_cocotb_test()                                      │
│    ├── make test-popcount                                       │
│    └── rtl/tb/test_common.py (Python test cases)                │
└─────────────────────────────────────────────────────────────────┘
```

## Files Reference

| File | Purpose |
|------|---------|
| `test_e2e_pipeline.py` | Main E2E test script |
| `rtl/common/*.v` | Core RTL modules |
| `rtl/tb/Makefile` | Cocotb test automation |
| `rtl/tb/test_common.py` | Python test cases |
| `model/conversion/quantize_ternary.py` | Ternary quantization |

## Next Steps

After verifying the E2E pipeline works:

1. **Run full test suite**: `cd rtl/tb && make sim`
2. **Synthesize for FPGA**: `cd fpga/xilinx && vivado -source synth_vivado.tcl`
3. **Run demo**: `python demo.py` (option 6 for E2E pipeline)

## Contributing

To add new tests:

1. Add test function to `rtl/tb/test_common.py`
2. Decorate with `@cocotb.test()`
3. Run: `make test-popcount`

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full guidelines.

---

*Last verified: August 2026 with Icarus Verilog 13.0, cocotb 2.0.1, Python 3.12*
