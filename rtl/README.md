# SiLens RTL Source Code

This directory contains the synthesizable Verilog RTL for the SiLens vision-language AI accelerator running SmolVLM-256M.

## Directory Structure

```
rtl/
├── common/                 # Shared utility modules
│   ├── popcount.v         # Population count for binary operations
│   ├── ternary_mac.v      # Ternary multiply-accumulate
│   ├── binary_dot_product.v  # XNOR + popcount dot product
│   ├── gelu_approx.v      # Piece-wise linear GELU approximation
│   ├── layer_norm.v       # Layer normalization
│   ├── softmax_approx.v   # Approximate softmax
│   └── rms_norm.v         # RMS normalization (for LLM)
├── vision_encoder/        # SigLIP-B/16 implementation (93M params)
│   ├── patch_embed.v      # Patch extraction + embedding (384x384 -> 576x768)
│   ├── vit_attention.v    # Multi-head self-attention (12 heads, 768 dim)
│   ├── vit_mlp.v          # MLP block (768->3072->768 with GELU)
│   ├── vit_block.v        # Complete transformer block
│   └── vision_encoder.v   # Full encoder (12 blocks + final LN)
├── projector/             # Multimodal projector
│   └── projector.v        # Linear projection (768->576)
├── language_model/        # SmolLM2-135M implementation (135M params)
│   ├── llm_attention.v    # Grouped-query attention with KV cache + RoPE
│   ├── llm_mlp.v          # SwiGLU MLP (576->1536->576)
│   ├── llm_block.v        # Decoder block (RMSNorm + Attention + MLP)
│   ├── llm_head.v         # LM head (RMSNorm + vocab projection)
│   └── language_model.v   # Full decoder (30 blocks + embeddings)
├── memory/                # On-chip memory subsystem
│   ├── activation_buffer.v # Double-buffered activation storage
│   ├── kv_cache.v         # Key-value cache for autoregressive decoding
│   ├── embedding_rom.v    # Token embedding lookup table
│   └── memory_controller.v # Arbiter for memory access
├── pcie/                  # PCIe interface modules
│   ├── pcie_wrapper.v     # Wrapper for vendor PCIe hard IP
│   ├── dma_engine.v       # DMA controller for host transfers
│   └── register_file.v    # Configuration and status registers
├── top/                   # Top-level integration
│   └── silens_top.v       # Complete SiLens accelerator
└── tb/                    # Testbenches
    ├── Makefile           # Cocotb testbench makefile
    └── test_common.py     # Tests for common modules
```

## Architecture Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Vision Encoder | Dimension | 768 |
| Vision Encoder | Layers | 12 |
| Vision Encoder | Heads | 12 |
| Vision Encoder | Image Size | 384x384 |
| Vision Encoder | Patch Size | 16x16 |
| Vision Encoder | Output Tokens | 576 |
| Projector | Input Dim | 768 |
| Projector | Output Dim | 576 |
| Language Model | Dimension | 576 |
| Language Model | Layers | 30 |
| Language Model | Heads | 9 |
| Language Model | MLP Dim | 1536 |
| Language Model | Vocabulary | 49,152 |
| Language Model | Max Context | 8,192 |
| Precision | Activation Width | 8 bits |
| Precision | Accumulator Width | 32 bits |


## Weight Encoding

Ternary weights (-1, 0, +1) are encoded in 2 bits:
- `00` = 0 (zero)
- `01` = +1 (add activation)
- `10` = -1 (subtract activation)
- `11` = reserved

This eliminates multipliers - MAC operations become add/subtract based on weight value.

## Module Interfaces

All modules use AXI-Stream-like valid/ready handshaking:

```verilog
// Input interface
input  wire [DATA_WIDTH-1:0]  data_in,
input  wire                   valid_in,
output wire                   ready_in,

// Output interface
output reg  [DATA_WIDTH-1:0]  data_out,
output reg                    valid_out,
input  wire                   ready_out
```

## Coding Guidelines

- Clock: `posedge clk`
- Reset: Active-low synchronous `rst_n`
- Naming: lowercase_with_underscores
- Parameters: UPPERCASE
- Target frequency: 100-200 MHz

## Running Testbenches

Each module includes an inline testbench (enabled with `ifdef SIMULATION):

```bash
# Using Icarus Verilog
iverilog -DSIMULATION -o tb_ternary_mac rtl/common/ternary_mac.v
vvp tb_ternary_mac

# Using Verilator
verilator --cc --exe -DSIMULATION rtl/common/ternary_mac.v
```

Or using cocotb:
```bash
cd rtl/tb
make test_all
```

## Key Design Decisions

1. **Ternary Weights**: Weights quantized to {-1, 0, +1} eliminate multipliers
2. **Sequential Block Sharing**: Single transformer block reused across layers to save area
3. **KV Cache**: Language model maintains key-value cache for autoregressive generation
4. **Piece-wise Linear Activations**: GELU/SiLU approximated with linear segments
5. **Newton-Raphson**: Inverse sqrt for normalization using iterative approximation

## Synthesis Notes

Target: SkyWater SKY130 (130nm)
- Die size: ~800mm²
- Clock: 100-200 MHz
- Power: 25W total
