---
layout: default
title: Architecture
permalink: /architecture/
---

<section class="page-header-section">
  <h1>System Architecture</h1>
  <p>How SiLens implements a vision-language model in hardwired silicon.</p>
</section>

<div class="content-wrapper">

## Overview

SiLens implements SmolVLM-256M as a single ASIC with three main components:

- **Vision Encoder** (SigLIP-B/16) — Processes images into visual tokens
- **Multimodal Projector** — Maps vision space to language space
- **Language Model** (SmolLM2-135M) — Generates text responses

---

## Model Components

### Vision Encoder (SigLIP-B/16)

- **Parameters:** 93M
- **Patch size:** 16×16 pixels
- **Input size:** 384×384
- **Hidden dimension:** 768
- **Layers:** 12 transformer blocks
- **Attention heads:** 12
- **Output:** 576 tokens

### Multimodal Projector

- **Parameters:** 18M
- **Input dimension:** 768
- **Output dimension:** 576
- **Type:** Linear projection

### Language Model (SmolLM2-135M)

- **Parameters:** 135M
- **Hidden dimension:** 576
- **Layers:** 30 transformer blocks
- **Attention heads:** 9
- **Vocabulary:** 49,152 tokens
- **Max context:** 8,192 tokens

---

## Hardwired Weight Encoding

The key innovation: model weights become physical wire connections.

### Ternary Quantization

SmolVLM-256M uses ternary weights {-1, 0, +1}:

- **Weight = +1** → Metal trace to VDD (power)
- **Weight = -1** → Metal trace to GND (ground)
- **Weight = 0** → No connection (implicit zero)

### Why This Works

**Traditional approach:** Weights stored in memory, loaded for each computation. Memory bandwidth becomes the bottleneck.

**SiLens approach:** Weights ARE the circuit. No memory access needed. Computation happens at wire speed (nanoseconds).

---

## Data Flow

1. **Image Input** — 384×384×3 RGB image
2. **Patch Extraction** — Split into 24×24 patches of 16×16 pixels
3. **Patch Embedding** — Convert to 576 tokens × 768 dimensions
4. **Vision Transformer** — Process through 12 layers
5. **Projection** — Map from 768 to 576 dimensions
6. **Concatenation** — Combine with text tokens
7. **Language Model** — Process through 30 layers
8. **Token Generation** — Autoregressive output
9. **Output Text** — Final response

---

## Physical Design

### Die Specifications

- **Process:** SkyWater SKY130 (130nm)
- **Die size:** ~800mm²
- **Metal layers:** 5
- **Core voltage:** 1.8V
- **I/O voltage:** 3.3V
- **Clock:** 100-200 MHz

### Area Breakdown

- **Vision encoder:** 280mm² (35%)
- **Language model:** 400mm² (50%)
- **Projector:** 55mm² (7%)
- **PCIe + I/O:** 40mm² (5%)
- **Power/clocking:** 25mm² (3%)

### Power Budget

- **Vision encoder:** 8W
- **Language model:** 12W
- **PCIe PHY:** 2W
- **Clock/control:** 2W
- **Margin:** 1W
- **Total:** 25W

---

## PCIe Interface

### Specifications

- **Standard:** PCIe 3.0
- **Lanes:** x4
- **Bandwidth:** 4 GB/s bidirectional
- **Power:** Slot-powered (75W max available)

### Register Map

- **0x000 CTRL** — Control register
- **0x004 STATUS** — Status/interrupt register
- **0x008 IMG_ADDR** — Image buffer DMA address
- **0x00C IMG_SIZE** — Image dimensions
- **0x010 OUT_ADDR** — Output buffer DMA address
- **0x014 OUT_LEN** — Output length
- **0x100 DMA_CTRL** — DMA control
- **0x200+ DEBUG** — Debug registers

---

## Performance Targets

- **Single-image latency:** <5ms
- **Throughput (pipelined):** 200+ images/sec
- **Token generation:** 50+ tokens/sec
- **Power efficiency:** 8+ images/joule

---

<div style="text-align: center; margin-top: 3rem;">
<a href="{{ site.baseurl }}/getting-started/" class="btn btn-primary">Get Started →</a>
<a href="{{ site.baseurl }}/docs/" class="btn btn-outline">View Documentation →</a>
</div>

</div>
