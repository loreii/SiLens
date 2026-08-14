# SiLens FPGA Feasibility Analysis

## Executive Summary

**TL;DR: The full SiLens design CANNOT fit on any single commercially available FPGA.** However, a reduced/partitioned version targeting specific use cases is feasible.

---

## Design Parameters Analysis

### Model Specifications (from RTL)

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Vision Encoder** | Dimension | 768 |
| | Layers | 12 |
| | Heads | 12 |
| | Parameters | ~93M |
| **Language Model** | Dimension | 576 |
| | Layers | 30 |
| | Heads | 9 |
| | MLP Dim | 1536 |
| | Vocab Size | 49,152 |
| | Max Seq Len | 8,192 |
| | Parameters | ~135M |
| **Projector** | Parameters | ~18M |
| **Total** | Parameters | ~246M |

### Weight Storage Requirements

With ternary quantization (1.58 bits effective, stored as 2 bits):

| Component | Weights | Storage (2-bit) |
|-----------|---------|-----------------|
| Vision Encoder | 93M | 23.25 MB |
| Language Model | 135M | 33.75 MB |
| Projector | 18M | 4.5 MB |
| **Total Weights** | **246M** | **~61.5 MB** |

### Activation/Buffer Requirements

| Buffer | Size | Notes |
|--------|------|-------|
| KV Cache (per layer) | 576 × 8192 × 8 bits × 2 | 75.5 MB per layer |
| KV Cache (30 layers) | - | **2.27 GB** |
| Activation buffers | ~576 × 4 × 8 bits | Negligible |
| Attention scores | 8192 × 8 bits | 64 KB per head |

---

## FPGA Resource Analysis

### Target FPGA Comparison

| FPGA | Logic Cells | DSPs | BRAM | Price |
|------|-------------|------|------|-------|
| **Xilinx Artix-7 200T** | 215K | 740 | 13.14 Mb | ~$300 |
| **Xilinx Kintex-7 325T** | 326K | 840 | 16.02 Mb | ~$1,500 |
| **Xilinx Kintex UltraScale+ KU5P** | 522K | 1,968 | 38.6 Mb | ~$4,000 |
| **Intel Arria 10 GX 660** | 660K | 1,687 | 42.6 Mb | ~$5,000 |
| **Xilinx Virtex UltraScale+ VU9P** | 2.6M | 6,840 | 345.6 Mb | ~$20,000+ |

### Resource Requirements Estimation

#### 1. Weight Storage

**Full Model:** 61.5 MB = 492 Mb
- **Artix-7 200T:** Has 13.14 Mb → Only 2.7% of needed storage ❌
- **Kintex-7 325T:** Has 16.02 Mb → Only 3.3% of needed storage ❌
- **VU9P:** Has 345.6 Mb → Still only 70% of needed storage ❌

**Conclusion:** Weights MUST use external memory (DDR4/HBM)

#### 2. KV Cache

**Full 8K context:** 2.27 GB = 18.16 Gb
- **No FPGA** has this much on-chip memory
- **Must use external DDR4/DDR5 or HBM**

**Reduced 256-token context:** 
- Per layer: 576 × 256 × 8 × 2 = 2.36 Mb
- 30 layers: 70.8 Mb
- **Fits in VU9P** but not smaller FPGAs

#### 3. Compute Units (DSPs)

For a **single token inference** with parallelism=16:

| Operation | MACs per token | DSPs needed (theoretical) |
|-----------|----------------|---------------------------|
| Q/K/V projection | 576 × 576 × 3 | ~100 DSPs |
| Attention score | 576 × seq_len | Variable |
| Output projection | 576 × 576 | ~35 DSPs |
| MLP (gate+up+down) | 576 × 1536 × 3 | ~275 DSPs |
| **Per layer** | ~3.5M MACs | ~400 DSPs |
| **30 layers** | ~105M MACs | N/A (time-multiplexed) |

With time-multiplexing and PARALLEL=16:
- **Minimum DSPs:** ~64 (for basic operation)
- **Optimal DSPs:** ~200-400 (for reasonable throughput)

---

## Feasibility Scenarios

### ❌ Scenario 1: Full Model on Single FPGA (INFEASIBLE)

**Requirements:**
- 492 Mb weight storage
- 18 Gb KV cache  
- Complex multi-head attention

**Reality:** No FPGA exists with sufficient on-chip memory.

### ⚠️ Scenario 2: FPGA + External DDR4 (PARTIALLY FEASIBLE)

**Architecture:**
```
┌─────────────────────────────────────────────┐
│                   FPGA                       │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Compute │  │ Control  │  │ DDR4     │   │
│  │ Engine  │  │ FSM      │  │ Controller│   │
│  │ (PE×16) │  │          │  │          │   │
│  └─────────┘  └──────────┘  └──────────┘   │
│       ↑            ↑              ↑         │
└───────┼────────────┼──────────────┼─────────┘
        │            │              │
        └────────────┼──────────────┘
                     ↓
              ┌──────────────┐
              │  DDR4 DRAM   │
              │  (4-8 GB)    │
              │  Weights +   │
              │  KV Cache    │
              └──────────────┘
```

**Challenges:**
- **Memory bandwidth bottleneck:** DDR4 ~25 GB/s vs needed ~50+ GB/s for real-time
- **Latency:** ~100ns per access vs ~1ns for BRAM
- **Throughput:** ~0.1-0.5 tokens/second (10-50x slower than target)

**Target FPGAs:**
- Xilinx Kintex UltraScale+ with DDR4
- Intel Arria 10 with DDR4

**Estimated Performance:** 
- ~200-500ms per token (vision + text)
- Interactive use: Marginal
- Batch processing: Acceptable

### ✅ Scenario 3: Reduced Model for FPGA Demo (FEASIBLE)

**Reduced Specifications:**
| Parameter | Full | Reduced | Storage |
|-----------|------|---------|---------|
| LLM Layers | 30 | 4 | 8× smaller |
| LLM Dim | 576 | 256 | 5× smaller |
| Max Seq Len | 8192 | 256 | 32× smaller |
| Vision Layers | 12 | 2 | 6× smaller |
| Total Weights | 246M | ~8M | 2 MB |
| KV Cache | 2.27 GB | ~2 MB | On-chip |

**This fits on Kintex-7!**

Resource estimate for reduced model:
- BRAM: ~14 Mb (weights + KV + buffers) ≤ 16 Mb ✓
- DSPs: ~200 ≤ 840 ✓
- Logic: ~150K LUTs ≤ 326K ✓

### ✅ Scenario 4: Accelerator Co-Processor (RECOMMENDED)

**Best practical approach:**

```
┌────────────┐      PCIe 3.0 x4      ┌──────────────┐
│   Host     │◄────────────────────►│    FPGA      │
│   CPU      │     (4 GB/s)         │  Accelerator │
│            │                       │              │
│  - Control │                       │ - Ternary    │
│  - Weights │                       │   MACs       │
│  - Memory  │                       │ - Attention  │
│            │                       │ - Softmax    │
└────────────┘                       └──────────────┘
```

**FPGA handles:**
- Ternary matrix-vector multiplication (compute-bound)
- Attention score computation
- Approximate softmax

**Host CPU handles:**
- Weight management (stream to FPGA)
- KV cache management (large memory)
- Token embedding/decoding

**Benefits:**
- Leverages FPGA's ternary compute efficiency
- Uses host's large memory for weights/KV
- Achievable on Artix-7 or Kintex-7

---

## Recommended Development Path

### Phase 1: Proof of Concept (Artix-7)
- **Target:** Artix-7 A200T ($300)
- **Scope:** Single transformer block, small dimensions
- **Goal:** Validate ternary MAC efficiency, pipeline design

### Phase 2: Reduced Model Demo (Kintex-7)
- **Target:** Kintex-7 K325T (~$1,500)
- **Scope:** 4-layer LLM, 256-dim, 256-token context
- **Goal:** End-to-end inference demonstration

### Phase 3: Full Accelerator (Kintex UltraScale+)
- **Target:** KU5P + DDR4 (~$5,000 board)
- **Scope:** Full model with external memory
- **Goal:** Practical performance benchmarks

### Phase 4: Production (ASIC)
- **Target:** Custom silicon
- **Scope:** Full 246M parameter model
- **Goal:** Commercial product

---

## RTL Changes Required for FPGA

### 1. Parameterizable Dimensions
Current RTL has fixed parameters. Need to make these synthesis-time configurable:

```verilog
// Add to silens_top.v
`ifdef FPGA_DEMO
    parameter LLM_LAYERS = 4;
    parameter LLM_DIM = 256;
    parameter MAX_SEQ_LEN = 256;
`else
    parameter LLM_LAYERS = 30;
    parameter LLM_DIM = 576;
    parameter MAX_SEQ_LEN = 8192;
`endif
```

### 2. External Memory Interface
Need to add DDR4 controller wrapper:

```verilog
module weight_loader #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 512  // DDR4 burst width
)(
    // AXI4 interface to DDR4 controller
    input  wire [DATA_WIDTH-1:0] ddr_rdata,
    input  wire                  ddr_rvalid,
    output wire [ADDR_WIDTH-1:0] ddr_raddr,
    output wire                  ddr_ren,
    
    // Internal weight interface
    output reg  [DIM*2-1:0]      weight_row,
    output reg                   weight_valid
);
```

### 3. Layer Time-Multiplexing
Current RTL instantiates dedicated blocks. Need time-shared approach:

```verilog
// Instead of 30 separate llm_block instances:
// Use single instance + layer counter
module llm_layers_shared #(...)(
    input  wire [$clog2(NUM_LAYERS)-1:0] layer_idx,
    // ... single llm_block instance reused for all layers
);
```

### 4. Reduced Precision Option
Add 4-bit activation mode for smaller FPGAs:

```verilog
parameter ACT_WIDTH = 8;  // Change to 4 for demo
```

---

## Conclusion

| Target | Feasible? | Notes |
|--------|-----------|-------|
| Full model on FPGA | ❌ No | Impossible - need 61 MB weights + 2 GB KV |
| Reduced demo model | ✅ Yes | 4-layer, 256-dim fits Kintex-7 |
| Accelerator card | ✅ Yes | FPGA compute + host memory |
| ASIC | ✅ Yes | Original target, needs ~31 mm² |

**Recommendation:** Develop a **reduced demo model** for FPGA validation, then target **ASIC** for production. The FPGA demo proves the architecture works, validates ternary efficiency, and builds investor confidence before the larger ASIC investment.

---

## Appendix: Detailed BRAM Calculations

### Kintex-7 325T Available Resources
- 445 Block RAMs (36Kb each) = 16.02 Mb
- 890 half-block RAMs (18Kb each)

### Reduced Model Allocation
| Component | Size | BRAMs (36Kb) |
|-----------|------|--------------|
| LLM weights (4 layers) | 4 × (256×256×3 + 256×512×3) × 2 bits | ~50 |
| KV cache (256 tokens) | 256 × 256 × 8 × 2 × 4 layers | ~20 |
| Activation buffers | 256 × 8 × 4 | ~2 |
| Vision weights (2 layers) | 2 × 768 × 768 × 2 bits | ~80 |
| Softmax LUT | 1K entries | ~1 |
| **Total** | | **~153 / 445** |

**Utilization: 34%** - Leaves room for PCIe, control logic, debug.
