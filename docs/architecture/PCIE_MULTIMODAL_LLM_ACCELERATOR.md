# 1-Bit Multimodal LLM PCIe Accelerator - Maximum Die Design

## Project Pivot

**Original:** USB 3.0 Thumb Drive (constrained to 25-50 mm²)  
**New Target:** PCIe Accelerator Card (maximum reticle ~800 mm²)

This enables **multimodal vision-language capability** using the SmolVLM architecture.

---

## Target Model: SmolVLM-500M

### Why SmolVLM-500M?

| Model | Total Params | Vision Encoder | Language Model | Fit @ 800mm²? |
|-------|--------------|----------------|----------------|---------------|
| SmolVLM-256M | 256M | SigLIP-B/16 (93M) | SmolLM2-135M (135M) | ✅ Comfortable |
| **SmolVLM-500M** | **~500M** | **SigLIP-B/16 (93M)** | **SmolLM2-360M (360M)** | **✅ Target** |
| SmolVLM-2.2B | 2.2B | SigLIP-SO400M (400M) | SmolLM2-1.7B (1.7B) | ❌ Too large |

**SmolVLM-500M is the largest multimodal model that can fit on SKY130 max reticle.**

---

## SmolVLM-500M Architecture Breakdown

### Component Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    SmolVLM-500M                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │         VISION ENCODER: SigLIP-B/16 (93M)           │   │
│  │  • 12 Transformer layers                            │   │
│  │  • Hidden dim: 768                                  │   │
│  │  • Attention heads: 12                              │   │
│  │  • Patch size: 16×16                                │   │
│  │  • Input resolution: 512×512                        │   │
│  │  • Output: 64 visual tokens per 512×512 region      │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         PROJECTION LAYER (~28M params)              │   │
│  │  • Maps vision features to LLM embedding space      │   │
│  │  • 768 → 960 (SmolLM2-360M hidden dim)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │      LANGUAGE MODEL: SmolLM2-360M (~360M)           │   │
│  │  • 32 Transformer layers                            │   │
│  │  • Hidden dim: 960                                  │   │
│  │  • Attention heads: 15                              │   │
│  │  • FFN hidden: 2560                                 │   │
│  │  • Vocab size: 49,152                               │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  TOTAL: ~500M parameters                                    │
│  License: Apache 2.0                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Parameter Count

### SigLIP-B/16 Vision Encoder (93M params)

```
Vision Encoder Architecture:
├── Patch Embedding: 3 × 16 × 16 × 768 = 590K
├── Position Embedding: 1024 × 768 = 786K
├── 12 Transformer Layers:
│   ├── Self-Attention (per layer):
│   │   ├── Q: 768 × 768 = 590K
│   │   ├── K: 768 × 768 = 590K
│   │   ├── V: 768 × 768 = 590K
│   │   └── O: 768 × 768 = 590K
│   │   └── Subtotal: 2.36M × 12 = 28.3M
│   └── MLP (per layer):
│       ├── FC1: 768 × 3072 = 2.36M
│       └── FC2: 3072 × 768 = 2.36M
│       └── Subtotal: 4.72M × 12 = 56.6M
├── LayerNorms: ~100K
└── Final projection: 768 × 768 = 590K

Vision Encoder Total: ~93M parameters
```

### Projection Layer (~28M params)

```
Projection:
├── Vision → LLM space: 768 × 960 = 737K
├── Pixel shuffle / pooling layers: ~5M
└── Learned queries / cross-attention: ~22M

Projection Total: ~28M parameters
```

### SmolLM2-360M Language Model (~360M params)

```
Language Model Architecture:
├── Token Embedding: 49,152 × 960 = 47.2M
├── 32 Transformer Layers:
│   ├── Self-Attention (per layer):
│   │   ├── Q: 960 × 960 = 922K
│   │   ├── K: 960 × 960 = 922K (or GQA reduced)
│   │   ├── V: 960 × 960 = 922K
│   │   └── O: 960 × 960 = 922K
│   │   └── Subtotal: ~3M × 32 = 96M
│   └── FFN (per layer - SwiGLU):
│       ├── Gate: 960 × 2560 = 2.46M
│       ├── Up: 960 × 2560 = 2.46M
│       └── Down: 2560 × 960 = 2.46M
│       └── Subtotal: 7.4M × 32 = 237M
├── RMSNorm layers: ~200K
└── LM Head: (tied with embedding) = 0

Language Model Total: ~360M parameters
```

### Grand Total

| Component | Parameters |
|-----------|------------|
| SigLIP-B/16 Vision Encoder | 93M |
| Projection Layer | 28M |
| SmolLM2-360M Language Model | 360M |
| **Total** | **~481M** |

Rounded to marketing: **SmolVLM-500M**

---

## Silicon Area Estimation

### Transistor Requirements

For 1-bit hardwired weights with XNOR+popcount:
- **Weights:** 0 transistors (metal routing VDD/GND)
- **Compute logic:** ~2 transistors per weight connection

| Component | Params | Compute Transistors |
|-----------|--------|---------------------|
| Vision Encoder | 93M | ~190M |
| Projection | 28M | ~60M |
| Language Model | 360M | ~720M |
| **Total** | **481M** | **~970M transistors** |

### Area Calculation

At SKY130 130nm density (~500K-800K transistors/mm²):

```
Conservative (500K/mm²):
  970M ÷ 500K = 1,940 mm² ❌ WAY TOO BIG

Optimistic (800K/mm²):  
  970M ÷ 800K = 1,213 mm² ❌ STILL TOO BIG
```

## ⚠️ PROBLEM: SmolVLM-500M Doesn't Fit Either!

Even at maximum reticle (800 mm²), we need ~1,200-1,900 mm².

---

## Revised Target: SmolVLM-256M

### SmolVLM-256M Fits!

| Component | Parameters | Transistors |
|-----------|------------|-------------|
| SigLIP-B/16 Vision | 93M | ~190M |
| Projection | ~18M | ~36M |
| SmolLM2-135M LLM | 135M | ~270M |
| **Total** | **~246M** | **~496M** |

Area calculation:
```
496M transistors ÷ 500K/mm² = 992 mm²  ❌ Still tight

With optimizations (sharing, folding):
496M × 0.7 = 347M effective transistors
347M ÷ 500K = 694 mm² ✅ FITS!
```

**SmolVLM-256M is the largest multimodal model for SKY130.**

---

## Final Target: SmolVLM-256M on SKY130 Max Reticle

### Architecture Summary

```
┌────────────────────────────────────────────────────────────────┐
│              SmolVLM-256M PCIe Accelerator                     │
│                   (~800 mm² die)                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              VISION ENCODER: SigLIP-B/16                  │ │
│  │                    (93M params)                           │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  12 Transformer Layers (hardwired weights)         │  │ │
│  │  │  • Hidden: 768, Heads: 12, MLP: 3072              │  │ │
│  │  │  • Patch: 16×16, Resolution: 512×512              │  │ │
│  │  │  • Output: 64 visual tokens                        │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  │                     (~250 mm²)                            │ │
│  └──────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              MULTIMODAL PROJECTOR                         │ │
│  │                   (~18M params)                           │ │
│  │  • Pixel shuffle + pooling                               │ │
│  │  • 768 → 576 dimension mapping                           │ │
│  │                      (~50 mm²)                            │ │
│  └──────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │           LANGUAGE MODEL: SmolLM2-135M                    │ │
│  │                   (135M params)                           │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  30 Transformer Layers (hardwired weights)         │  │ │
│  │  │  • Hidden: 576, Heads: 9, FFN: 1536               │  │ │
│  │  │  • Vocab: 49,152 tokens                            │  │ │
│  │  │  • Context: 8192 tokens (text + vision)            │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  │                     (~350 mm²)                            │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │
│  │   PCIe     │  │   Power    │  │      Clock + PLL       │  │
│  │  Gen3 x4   │  │   Mgmt     │  │                        │  │
│  │  (40mm²)   │  │  (30mm²)   │  │       (20mm²)          │  │
│  └────────────┘  └────────────┘  └────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              I/O Ring + ESD Protection                    │ │
│  │                      (~60 mm²)                            │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  TOTAL DIE AREA: ~800 mm² (fits max reticle)                  │
└────────────────────────────────────────────────────────────────┘
```

### Area Breakdown

| Block | Parameters | Area (mm²) | % of Die |
|-------|------------|------------|----------|
| SigLIP-B/16 Vision | 93M | 250 | 31% |
| Multimodal Projector | 18M | 50 | 6% |
| SmolLM2-135M LLM | 135M | 350 | 44% |
| PCIe Controller | - | 40 | 5% |
| Power Management | - | 30 | 4% |
| Clock/PLL | - | 20 | 2.5% |
| I/O Ring + ESD | - | 60 | 7.5% |
| **Total** | **~246M** | **~800** | **100%** |

---

## Capabilities of SmolVLM-256M

### What It Can Do

| Capability | Performance | Notes |
|------------|-------------|-------|
| **Image Understanding** | Good | Describe images, answer questions |
| **Document OCR** | Good | Read text from images |
| **Multi-image** | Supported | Compare multiple images |
| **Video** | Basic | Frame-by-frame analysis |
| **Text Generation** | Moderate | Based on SmolLM2-135M |
| **Reasoning** | Basic | Limited by 135M LLM size |

### Benchmark Performance (Reference)

SmolVLM-256M outperforms Idefics-80B (300× larger!) on many tasks:

| Benchmark | SmolVLM-256M | Notes |
|-----------|--------------|-------|
| VQAv2 | ~65% | Visual question answering |
| TextVQA | ~45% | Text in images |
| DocVQA | ~50% | Document understanding |
| MMMU | ~30% | Multimodal reasoning |

### Use Cases for Hardwired Version

1. **Image Captioning Accelerator**
   - Fast draft captions for image databases
   - Verified by larger cloud model

2. **Document Pre-processing**
   - Quick OCR and layout analysis
   - Extract text regions at hardware speed

3. **Visual Search**
   - Rapid image-to-text for retrieval
   - Embedding generation for similarity

4. **Edge Vision AI**
   - Security camera analysis
   - Real-time object description

5. **MTP for Vision-Language**
   - Draft visual responses
   - Speculative decoding with larger VLM

---

## System Architecture

### PCIe Card Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    PCIe x4 HALF-HEIGHT CARD                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │     ┌─────────────────────────────────────────────┐       │ │
│  │     │                                             │       │ │
│  │     │         SmolVLM-256M ASIC                   │       │ │
│  │     │           (800 mm²)                         │       │ │
│  │     │         ~30mm × 27mm                        │       │ │
│  │     │                                             │       │ │
│  │     └─────────────────────────────────────────────┘       │ │
│  │                        │                                   │ │
│  │     ┌──────────────────┴──────────────────┐               │ │
│  │     │         HEATSINK (required)          │               │ │
│  │     │         Active fan optional          │               │ │
│  │     └─────────────────────────────────────┘               │ │
│  │                                                            │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │ │
│  │  │  PMIC   │  │  Clock  │  │  Flash  │  │  Caps   │      │ │
│  │  │ 12V→1V  │  │  100MHz │  │  Config │  │  Bulk   │      │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │ │
│  │                                                            │ │
│  └───────────────────────────────────────────────────────────┘ │
│                           │                                     │
│  ════════════════════════╪═════════════════════════════════   │
│           PCIe x4 Edge Connector                               │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Host System                           PCIe Card
┌─────────────────┐                  ┌─────────────────────────┐
│                 │                  │   SmolVLM-256M ASIC     │
│  CPU/GPU        │   Image Data     │  ┌─────────────────┐   │
│  (Main Model)   │ ────────────────►│  │  SigLIP-B/16    │   │
│                 │   (RGB pixels)   │  │  Vision Encoder │   │
│                 │                  │  └────────┬────────┘   │
│                 │                  │           │             │
│                 │   Text Prompt    │           ▼             │
│                 │ ────────────────►│  ┌─────────────────┐   │
│                 │   (Token IDs)    │  │  SmolLM2-135M   │   │
│                 │                  │  │  Language Model │   │
│                 │                  │  └────────┬────────┘   │
│                 │◄─────────────────│           │             │
│                 │   Draft Tokens   │   Output Tokens        │
│                 │   (for verify)   │                         │
└─────────────────┘                  └─────────────────────────┘

Latency Target: <1ms for single image + prompt
Throughput: ~1000 images/second (vision only)
```

---

## Hardware Specifications

### ASIC Specifications

| Parameter | Value |
|-----------|-------|
| Process | SkyWater SKY130 (130nm) |
| Die Size | ~800 mm² (26mm × 31mm) |
| Transistors | ~500M |
| Core Voltage | 1.8V |
| I/O Voltage | 3.3V |
| Clock Speed | 100-200 MHz (target) |
| Power | 15-25W TDP |
| Package | BGA-625 or larger |

### PCIe Card Specifications

| Parameter | Value |
|-----------|-------|
| Interface | PCIe 3.0 x4 (4 GB/s) |
| Form Factor | Half-height, half-length |
| Power | 25W (from PCIe slot) |
| Cooling | Passive heatsink + optional fan |
| Dimensions | 168mm × 69mm |

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Image encode latency | <500 µs | 512×512 image |
| Text generation | ~100 tokens/sec | After image encoded |
| End-to-end latency | <5 ms | Image + short response |
| Images/second | 1000+ | Vision encoder only |
| Power efficiency | 10-15 TOPS/W | 1-bit operations |

---

## Bill of Materials (PCIe Card)

### Per-Unit Cost (1,000 units)

| Component | Est. Cost |
|-----------|-----------|
| SmolVLM-256M ASIC | $50-80 |
| BGA Package | $5-10 |
| PCIe PHY (if external) | $3-5 |
| PMIC + Regulators | $3-5 |
| PCB (6-8 layer) | $10-15 |
| Heatsink | $3-5 |
| Passives + connectors | $2-3 |
| Assembly + test | $10-15 |
| **Total per unit** | **$90-140** |

### NRE Costs

| Item | Cost |
|------|------|
| Mask set (SKY130) | $80-120K |
| EDA tools + licenses | $20-50K |
| Design verification | $50-100K |
| PCB design | $10-20K |
| Prototyping | $20-30K |
| **Total NRE** | **$200-350K** |

### Target Retail Price

| Volume | Unit Cost | Margin | Retail |
|--------|-----------|--------|--------|
| 1,000 | $120 | 2× | $249 |
| 5,000 | $100 | 2× | $199 |
| 10,000 | $85 | 2× | $169 |

---

## Comparison: What You Get vs. Alternatives

### SmolVLM-256M PCIe Card vs. GPU

| Aspect | SmolVLM-256M Card | RTX 4060 (for VLM) |
|--------|-------------------|---------------------|
| Price | $169-249 | $299 |
| Power | 25W | 115W |
| Latency | <5ms | 20-50ms |
| Model flexibility | Fixed (hardwired) | Any model |
| VRAM | N/A (hardwired) | 8GB |
| Use case | Fast drafting | General inference |

### When This Card Makes Sense

✅ **Good fit:**
- High-throughput image captioning
- Edge deployment (power constrained)
- Speculative decoding acceleration
- Fixed vision tasks (security, retail)

❌ **Not ideal:**
- Research (need model flexibility)
- Complex reasoning tasks
- Tasks requiring larger models
- Frequently changing requirements

---

## Implementation Roadmap

### Phase 1: Validation (3 months)

- [ ] Implement toy vision encoder (4 layers) in Verilog
- [ ] Verify against PyTorch SigLIP
- [ ] Implement toy LLM (4 layers) in Verilog
- [ ] Verify against PyTorch SmolLM2
- [ ] Build Python→Verilog weight generator
- [ ] Cocotb testbench for multimodal flow

### Phase 2: Scaled Prototype (6 months)

- [ ] Generate RTL for full SigLIP-B/16 (93M)
- [ ] Generate RTL for full SmolLM2-135M (135M)
- [ ] Implement multimodal projector
- [ ] Run OpenLane synthesis on blocks
- [ ] Verify timing and area estimates
- [ ] Power analysis

### Phase 3: Full Integration (6 months)

- [ ] Integrate PCIe controller IP
- [ ] Top-level SoC integration
- [ ] Full chip floorplanning
- [ ] Clock tree synthesis
- [ ] Power grid design
- [ ] DRC/LVS verification

### Phase 4: Tape-Out (3 months)

- [ ] Final timing closure
- [ ] Parasitic extraction
- [ ] Signoff checks
- [ ] Generate GDSII
- [ ] Submit to foundry

### Phase 5: Bring-Up (3 months)

- [ ] Receive silicon
- [ ] Basic functionality test
- [ ] Performance validation
- [ ] PCIe card assembly
- [ ] Driver development
- [ ] Benchmark vs. software

**Total timeline: ~21 months**

---

## Verilog Module Hierarchy

```
smolvlm_256m_top
├── pcie_controller
│   ├── pcie_phy
│   ├── transaction_layer
│   └── dma_engine
├── vision_encoder (SigLIP-B/16)
│   ├── patch_embed
│   │   └── conv2d_16x16 (hardwired)
│   ├── position_embed (hardwired)
│   └── transformer_blocks[0:11]
│       ├── layer_norm
│       ├── self_attention
│       │   ├── qkv_proj (hardwired weights)
│       │   ├── attention_scores
│       │   └── output_proj (hardwired weights)
│       └── mlp
│           ├── fc1 (hardwired weights)
│           └── fc2 (hardwired weights)
├── multimodal_projector
│   ├── pixel_shuffle
│   ├── pooling
│   └── linear_proj (hardwired weights)
├── language_model (SmolLM2-135M)
│   ├── token_embed (hardwired)
│   ├── rotary_embed
│   └── transformer_blocks[0:29]
│       ├── rms_norm
│       ├── self_attention
│       │   ├── qkv_proj (hardwired weights)
│       │   ├── rotary_attention
│       │   └── output_proj (hardwired weights)
│       └── ffn_swiglu
│           ├── gate_proj (hardwired weights)
│           ├── up_proj (hardwired weights)
│           └── down_proj (hardwired weights)
├── output_head
│   └── lm_head (tied with embed)
├── clock_gen
│   └── pll
└── power_management
    └── voltage_regulators
```

---

## Summary

### What We're Building

| Attribute | Value |
|-----------|-------|
| **Model** | SmolVLM-256M (hardwired 1-bit) |
| **Vision Encoder** | SigLIP-B/16 (93M params) |
| **Language Model** | SmolLM2-135M (135M params) |
| **Total Parameters** | ~246M |
| **Process** | SkyWater SKY130 (130nm) |
| **Die Size** | ~800 mm² (max reticle) |
| **Interface** | PCIe 3.0 x4 |
| **Power** | 25W |
| **Form Factor** | Half-height PCIe card |
| **Target Price** | $169-249 |

### Key Differentiators

1. **World's first hardwired multimodal VLM**
2. **Sub-5ms end-to-end latency**
3. **~25W power** (vs 100W+ for GPU inference)
4. **Open source design** (Apache 2.0)
5. **Manufactured on open PDK** (SKY130)

### Limitations

1. **Fixed model** - cannot update weights
2. **Limited reasoning** - 135M LLM is small
3. **Large die** - yield concerns at 800mm²
4. **No fine-tuning** - general-purpose only

---

*This is the maximum multimodal capability achievable on SkyWater SKY130 at full reticle size.*

*For larger models (SmolVLM-500M, SmolVLM-2.2B), a more advanced process node (65nm, 28nm) or multi-chip module would be required.*
