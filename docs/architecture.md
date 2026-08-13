---
layout: default
title: Architecture
---

<section class="page-header-section">
  <h1>System Architecture</h1>
  <p>A deep dive into how SiLens implements a vision-language model in hardwired silicon.</p>
</section>

<div class="content-wrapper">

## Architecture Overview

SiLens implements the complete SmolVLM-256M model as a single ASIC. The chip integrates:

- **Vision Encoder** (SigLIP-B/16) — Processes images into visual tokens
- **Multimodal Projector** — Maps vision space to language space  
- **Language Model** (SmolLM2-135M) — Generates text responses

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SiLens ASIC (~800mm²)                             │
│                                                                             │
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

### Vision Encoder (SigLIP-B/16)

| Parameter | Value |
|-----------|-------|
| Parameters | 93M |
| Patch size | 16×16 pixels |
| Input size | 384×384 |
| Hidden dim | 768 |
| Layers | 12 |
| Attention heads | 12 |
| Output tokens | 576 |

### Multimodal Projector

| Parameter | Value |
|-----------|-------|
| Parameters | 18M |
| Input dim | 768 |
| Output dim | 576 |
| Type | Linear projection |

### Language Model (SmolLM2-135M)

| Parameter | Value |
|-----------|-------|
| Parameters | 135M |
| Hidden dim | 576 |
| Layers | 30 |
| Attention heads | 9 |
| Vocabulary | 49,152 tokens |
| Max context | 8,192 tokens |

---

## Hardwired Weight Encoding

The key innovation in SiLens is encoding model weights as physical wire connections rather than storing them in memory.

### Ternary Quantization

SmolVLM-256M uses ternary weights: **{-1, 0, +1}**. In silicon:

| Weight | Physical Implementation |
|--------|------------------------|
| **+1** | Metal trace to VDD (power) |
| **-1** | Metal trace to GND (ground) |
| **0** | No connection (implicit zero) |

### Traditional vs Hardwired

**Traditional approach (memory-based):**
```verilog
wire [7:0] weight;        // 8-bit weight from SRAM
wire [7:0] activation;
wire [15:0] result = weight * activation;  // Multiply-accumulate
```

**SiLens approach (hardwired):**
```verilog
// Weight encoded as static wire connection
// For +1 weight:
assign result = activation;    // Pass through

// For -1 weight:
assign result = -activation;   // Negate

// For 0 weight:
assign result = 0;             // No contribution
```

### XNOR-Popcount for Binary Operations

```verilog
module binary_dot_product #(
    parameter WIDTH = 512
)(
    input  wire [WIDTH-1:0] activations,
    input  wire [WIDTH-1:0] weights,      // Hardwired
    output wire [$clog2(WIDTH):0] result
);
    wire [WIDTH-1:0] xnor_result;
    
    // XNOR: same bits → 1, different → 0
    assign xnor_result = ~(activations ^ weights);
    
    // Popcount: count 1s
    popcount #(.WIDTH(WIDTH)) pc (
        .in(xnor_result),
        .count(result)
    );
endmodule
```

---

## Data Flow Pipeline

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

---

## Physical Design

### Die Specifications

| Parameter | Target |
|-----------|--------|
| Process | SkyWater SKY130 (130nm) |
| Die size | ~800mm² |
| Metal layers | 5 |
| Core voltage | 1.8V |
| I/O voltage | 3.3V |
| Clock | 100-200 MHz |

### Area Breakdown

| Component | Area | Percentage |
|-----------|------|------------|
| Vision encoder | 280mm² | 35% |
| Language model | 400mm² | 50% |
| Projector | 55mm² | 7% |
| PCIe + I/O | 40mm² | 5% |
| Power/clocking | 25mm² | 3% |
| **Total** | **800mm²** | 100% |

### Power Budget

| Component | Power |
|-----------|-------|
| Vision encoder | 8W |
| Language model | 12W |
| PCIe PHY | 2W |
| Clock/control | 2W |
| Margin | 1W |
| **Total** | **25W** |

---

## PCIe Interface

### Specifications

| Parameter | Value |
|-----------|-------|
| Standard | PCIe 3.0 |
| Lanes | x4 |
| Bandwidth | 4 GB/s bidirectional |
| Power | Slot-powered (75W max) |

### Register Map

| Offset | Name | Description |
|--------|------|-------------|
| 0x000 | CTRL | Control register |
| 0x004 | STATUS | Status/interrupt register |
| 0x008 | IMG_ADDR | Image buffer DMA address |
| 0x00C | IMG_SIZE | Image dimensions |
| 0x010 | OUT_ADDR | Output buffer DMA address |
| 0x014 | OUT_LEN | Output length |
| 0x100 | DMA_CTRL | DMA control |
| 0x200+ | DEBUG | Debug registers |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Single-image latency | <5ms |
| Throughput (pipelined) | 200+ img/sec |
| Token generation | 50+ tokens/sec |
| Power efficiency | 8+ images/joule |

---

<div style="text-align: center; margin-top: 3rem;">
<a href="{{ site.baseurl }}/getting-started/" class="btn btn-primary">Get Started →</a>
<a href="{{ site.baseurl }}/docs/" class="btn btn-outline">View Documentation →</a>
</div>

</div>
