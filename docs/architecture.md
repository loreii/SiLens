---
layout: default
title: Architecture
permalink: /architecture/
---

# SiLens Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SiLens ASIC (~800mm²)                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    VISION ENCODER: SigLIP-B/16                        │ │
│  │                         (93M parameters)                               │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐            │ │
│  │  │ Patch   │  │ Embed   │  │ Trans-  │  ...  │ Trans-  │            │ │
│  │  │ Extract │→ │ Layer   │→ │ former  │→      │ former  │            │ │
│  │  │         │  │         │  │ Block 1 │       │ Block 12│            │ │
│  │  └─────────┘  └─────────┘  └─────────┘       └─────────┘            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    ↓                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    MULTIMODAL PROJECTOR (18M params)                  │ │
│  │                      768-dim → 576-dim mapping                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    ↓                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                  LANGUAGE MODEL: SmolLM2-135M                         │ │
│  │                        (135M parameters)                               │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐            │ │
│  │  │ Token   │  │ Trans-  │  │ Trans-  │  ...  │ Trans-  │→ LM Head  │ │
│  │  │ Embed   │→ │ former  │→ │ former  │→      │ former  │            │ │
│  │  │         │  │ Block 1 │  │ Block 2 │       │ Block 30│            │ │
│  │  └─────────┘  └─────────┘  └─────────┘       └─────────┘            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   PCIe 3.0  │  │    Power    │  │    Clock    │  │     I/O     │      │
│  │   x4 PHY    │  │  Management │  │   & Reset   │  │   Control   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Components

### 1. Vision Encoder (SigLIP-B/16)

| Parameter | Value |
|-----------|-------|
| Parameters | 93M |
| Patch size | 16×16 |
| Image size | 384×384 |
| Hidden dim | 768 |
| Layers | 12 |
| Heads | 12 |
| Output tokens | 576 |

### 2. Multimodal Projector

| Parameter | Value |
|-----------|-------|
| Parameters | 18M |
| Input dim | 768 |
| Output dim | 576 |
| Type | Linear projection |

### 3. Language Model (SmolLM2-135M)

| Parameter | Value |
|-----------|-------|
| Parameters | 135M |
| Hidden dim | 576 |
| Layers | 30 |
| Heads | 9 |
| Vocabulary | 49,152 |
| Context | 8,192 tokens |

---

## Hardwired Weight Encoding

### Ternary Weights (-1, 0, +1)

**Traditional (memory-based):**
```verilog
wire [7:0] weight;        // 8-bit weight from SRAM
wire [7:0] activation;
wire [15:0] result = weight * activation;  // Multiply-accumulate
```

**Hardwired (SiLens approach):**
```verilog
// Weight encoded as static wire connection:
//   +1 → wire connected to VDD
//   -1 → wire connected to GND  
//    0 → no connection

// For +1 weight:
assign result = activation;  // Just pass through

// For -1 weight:
assign result = -activation; // Negate (invert + 1)

// For 0 weight:
assign result = 0;           // No contribution
```

### XNOR-Popcount Implementation

For binary weights {-1, +1}:

```verilog
module binary_dot_product #(
    parameter WIDTH = 512
)(
    input  wire [WIDTH-1:0] activations,  // Binary activations
    input  wire [WIDTH-1:0] weights,       // Hardwired weights
    output wire [$clog2(WIDTH):0] result
);
    wire [WIDTH-1:0] xnor_result;
    
    // XNOR: same bits → 1, different bits → 0
    assign xnor_result = ~(activations ^ weights);
    
    // Popcount: count number of 1s
    // Result = 2 * popcount - WIDTH (to get range [-WIDTH, +WIDTH])
    popcount #(.WIDTH(WIDTH)) pc (
        .in(xnor_result),
        .count(result)
    );
endmodule
```

---

## Data Flow

### Inference Pipeline

```
1. Image Input (384×384×3 RGB)
   ↓
2. Patch Extraction (24×24 patches of 16×16)
   ↓
3. Patch Embedding (576 tokens × 768 dim)
   ↓
4. Vision Transformer (12 layers)
   ↓
5. Projection (768 → 576 dim)
   ↓
6. Concatenate with text tokens
   ↓
7. Language Model (30 layers)
   ↓
8. Token Generation (autoregressive)
   ↓
9. Output Text
```

### Memory Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Image input buffer | 442 KB | 384×384×3 bytes |
| Vision activations | 3.5 MB | 576×768×8 bytes (FP8) |
| LLM KV cache | 4 MB | For 2K context |
| Token embeddings | 113 MB | 49152×576×4 bytes |
| **Total on-chip** | **~130 MB** | Excluding hardwired weights |

---

## Physical Design Targets

### Die Specifications

| Parameter | Target |
|-----------|--------|
| Process | SkyWater SKY130 (130nm) |
| Die size | ~800mm² (max reticle) |
| Metal layers | 5 |
| Core voltage | 1.8V |
| I/O voltage | 3.3V |
| Clock | 100-200 MHz |

### Power Budget

| Component | Power |
|-----------|-------|
| Vision encoder | 8W |
| Language model | 12W |
| PCIe PHY | 2W |
| Clock/control | 2W |
| Margin | 1W |
| **Total** | **25W** |

### Area Breakdown (Estimated)

| Component | Area | % |
|-----------|------|---|
| Vision encoder | 280mm² | 35% |
| Language model | 400mm² | 50% |
| Projector | 55mm² | 7% |
| PCIe + I/O | 40mm² | 5% |
| Power/clocking | 25mm² | 3% |
| **Total** | **800mm²** | 100% |

---

## Interface

### PCIe 3.0 x4

| Parameter | Value |
|-----------|-------|
| Bandwidth | 4 GB/s (bidirectional) |
| Lanes | 4 |
| Width | x4 slot |
| Power | Slot-powered (75W max) |

### Register Map

| Offset | Name | Description |
|--------|------|-------------|
| 0x000 | CTRL | Control register |
| 0x004 | STATUS | Status register |
| 0x008 | IMG_ADDR | Image buffer address |
| 0x00C | IMG_SIZE | Image dimensions |
| 0x010 | OUT_ADDR | Output buffer address |
| 0x014 | OUT_LEN | Output length |
| 0x100 | DMA_CTRL | DMA control |
| 0x200+ | DEBUG | Debug registers |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Latency (single image) | <5ms |
| Throughput (pipelined) | 200+ img/sec |
| Token generation | 50+ tokens/sec |
| Power efficiency | 8+ img/joule |

---

[← Back to Home]({{ site.baseurl }}/)
