# SiLens Development Plan

> **Last Updated:** August 19, 2026  
> **Status:** Active Development  
> **Version:** 0.2.0

This document outlines the development roadmap for the SiLens open-source hardwired vision-language AI accelerator.

---

## Product Variants

SiLens now supports multiple hardware variants from a shared codebase:

| Variant | Die Size | Target | Status |
|---------|----------|--------|--------|
| **SiLens VLM** | 800mm² | Full Vision-Language Model | 🟢 Complete |
| **SiLens Edge** | 50mm² | Ultra-fast Edge Classifier | 🟢 Complete |

### Variant Architecture

```
Shared Components (openlane/level1, level2, rtl/, sdk/, drivers/)
    │
    ├── variants/silens-vlm/      # 800mm² VLM SoC
    │   ├── openlane/level3/      # VLM-specific subsystems
    │   └── openlane/level4/      # VLM top integration
    │
    └── variants/silens-edge/     # 50mm² Edge Classifier
        ├── openlane/level3/      # Edge-specific blocks
        └── openlane/level4/      # Edge top integration
```

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Architecture specification | 🟢 Complete | 800mm² SoC design finalized |
| RTL design (Verilog) | 🟢 Complete | Core modules verified |
| **SoC Top-Level Design** | 🟢 **Complete** | silens_soc.v for 800mm² target |
| **DDR3 Controller** | 🟢 **Complete** | DDR3-1066 x32 interface |
| **Host Interface** | 🟢 **Complete** | Parallel bus + CDC FIFOs |
| **VLM Core** | 🟢 **Complete** | Vision + Projector + LLM integrated |
| **OpenLane Config** | 🟢 **Complete** | 800mm² synthesis setup |
| **Level 1-3 Hierarchy** | 🟢 **Complete** | 15 blocks RTL & configs ready |
| **SiLens Edge Variant** | 🟢 **Complete** | 50mm² QFN-48 classifier |
| RTL simulation (E2E pipeline) | 🟢 Complete | Icarus Verilog + cocotb |
| Model quantization tools | 🟢 Complete | Ternary quantization working |
| Semantic equivalence tests | 🟢 Complete | 94% visual, 87.9% weight similarity |
| FPGA prototype | 🔴 Not started | **Next priority** |
| Physical design (ASIC) | 🟡 In progress | OpenLane config ready |
| PCB design | 🔴 Not started | Depends on FPGA validation |
| SDK/Drivers | 🟡 In progress | Basic structure exists |
| Documentation | 🟢 Complete | E2E guides, architecture docs |

### Key Achievements

- ✅ End-to-end RTL simulation pipeline verified
- ✅ Ternary quantization preserves 99.7% activation similarity
- ✅ Visual understanding validated with Lenna test (94% similarity)
- ✅ Comprehensive documentation for reproducibility
- ✅ **800mm² SoC architecture designed for SkyWater SKY130**
- ✅ **DDR3 memory controller for external KV cache**
- ✅ **Parallel host interface with FPGA bridge support**
- ✅ **OpenLane synthesis configuration for full-custom fabrication**
- ✅ **Confirmed SkyWater 26×32mm reticle supports 800mm² single-shot**
- ✅ **SiLens Edge variant (50mm²) complete with NanoViT + Classifier**

---

## SiLens Edge Specifications

The Edge variant targets embedded/industrial applications:

| Specification | Value |
|---------------|-------|
| Die Size | 50mm² (7mm × 7mm) |
| Process | SkyWater SKY130 130nm |
| Clock | 200MHz |
| Power | 3W TDP |
| Package | QFN-48 |
| **Model** | TinyVLM-20M |
| - Vision Encoder | NanoViT-12M (6 layers, 192-dim, 3 heads) |
| - Classifier | 7M params (4 layers, 128-dim) |
| **Performance** | |
| - Latency | <1ms per inference |
| - Throughput | 1000 FPS |
| - Active Power | <500mW |
| **Interfaces** | |
| - Primary | SPI Slave (50MHz) |
| - Config | I2C Slave |
| - GPIO | 8 pins (trigger, class output, status) |

**SiLens Edge Level 3 Blocks:**

| Block | Size | Function | Status |
|-------|------|----------|--------|
| vision_nano | ~15mm² | NanoViT-12M encoder | 🟢 Complete |
| classifier_head | ~10mm² | 4-layer MLP classifier | 🟢 Complete |
| io_edge | ~5mm² | SPI/I2C/GPIO interfaces | 🟢 Complete |
| sram_256kb | ~10mm² | Activation buffer (dual-port) | 🟢 Complete |

**SiLens Edge Level 4:**

| Block | Size | Function | Status |
|-------|------|----------|--------|
| silens_edge_soc | 50mm² | Top integration | 🟢 Complete |

---

## Development Roadmap

### Phase 1: FPGA Prototyping (High Priority)

**Goal:** Validate RTL design on actual hardware before ASIC fabrication.

**Timeline:** 4-6 weeks

| Task | Description | Status |
|------|-------------|--------|
| 1.1 | Synthesize for Xilinx Artix-7/Kintex-7 | 🔴 Not started |
| 1.2 | Map memory interfaces to FPGA block RAM | 🔴 Not started |
| 1.3 | Implement PCIe or USB interface | 🔴 Not started |
| 1.4 | Load actual quantized model weights | 🔴 Not started |
| 1.5 | Run real inference end-to-end | 🔴 Not started |
| 1.6 | Measure latency and throughput | 🔴 Not started |
| 1.7 | Create demo video | 🔴 Not started |

**Deliverables:**
- [ ] Working FPGA bitstream
- [ ] Performance benchmarks (latency, throughput, power)
- [ ] Demo application with live inference
- [ ] Synthesis reports (utilization, timing)

**Why this is critical:**
- Proves the design works in real hardware
- Identifies timing and integration issues before expensive ASIC fab
- Creates a demo platform for investors/Kickstarter
- Provides development platform for SDK/drivers

---

### Phase 2: SDK & Driver Development (High Priority)

**Goal:** Enable software integration for hardware interaction.

**Timeline:** 3-4 weeks (parallel with Phase 1)

| Task | Description | Status |
|------|-------------|--------|
| 2.1 | Complete Linux kernel driver (`silens_drv.c`) | 🟡 In progress |
| 2.2 | Implement PCIe/USB communication layer | 🔴 Not started |
| 2.3 | Complete Python SDK bindings | 🟡 In progress |
| 2.4 | Add model loading API | 🔴 Not started |
| 2.5 | Implement streaming inference | 🔴 Not started |
| 2.6 | Create example applications | 🔴 Not started |
| 2.7 | Write SDK documentation | 🔴 Not started |

**Deliverables:**
- [ ] Linux kernel module (loadable)
- [ ] Python package (`pip install silens`)
- [ ] Example: Image captioning script
- [ ] Example: Visual QA application
- [ ] API documentation

---

### Phase 3: Mixed-Precision Quantization (Medium Priority)

**Goal:** Improve model quality for critical layers identified in semantic equivalence testing.

**Timeline:** 2 weeks

| Task | Description | Status |
|------|-------------|--------|
| 3.1 | Implement per-layer precision configuration | 🔴 Not started |
| 3.2 | Keep `position_embedding` at INT8 | 🔴 Not started |
| 3.3 | Higher precision for attention K projections | 🔴 Not started |
| 3.4 | Higher precision for first/last LLM layers | 🔴 Not started |
| 3.5 | Re-run semantic equivalence tests | 🔴 Not started |
| 3.6 | Update RTL for mixed precision support | 🔴 Not started |

**Critical Layers Identified:**

| Layer | Current Cosine | Target | Action |
|-------|----------------|--------|--------|
| `position_embedding` | 0.53 | > 0.90 | Keep at INT8 |
| `text_model.layers.18.self_attn.k_proj` | 0.82 | > 0.90 | INT4 or INT8 |
| `text_model.layers.17.self_attn.k_proj` | 0.82 | > 0.90 | INT4 or INT8 |
| `text_model.layers.0.self_attn.q_proj` | 0.82 | > 0.90 | INT4 or INT8 |
| `lm_head` | 0.83 | > 0.90 | Consider INT8 |

**Expected Improvement:** Weight similarity from 87.9% → 92%+

---

### Phase 4: Benchmark on Standard Datasets (Medium Priority)

**Goal:** Quantify actual accuracy impact with industry-standard metrics.

**Timeline:** 2-3 weeks

| Task | Description | Status |
|------|-------------|--------|
| 4.1 | VQAv2 benchmark (Visual Question Answering) | 🔴 Not started |
| 4.2 | TextVQA benchmark (Text in images) | 🔴 Not started |
| 4.3 | COCO Captioning benchmark | 🔴 Not started |
| 4.4 | WikiText perplexity measurement | 🔴 Not started |
| 4.5 | Compare original vs quantized accuracy | 🔴 Not started |
| 4.6 | Document results for publication | 🔴 Not started |

**Target Metrics:**

| Benchmark | Original | Target (Quantized) | Acceptable Drop |
|-----------|----------|-------------------|-----------------|
| VQAv2 Accuracy | ~71% | > 67% | < 4% |
| TextVQA Accuracy | ~45% | > 42% | < 3% |
| COCO CIDEr | ~100 | > 90 | < 10% |
| WikiText PPL | ~15 | < 18 | < 20% |

---

### Phase 5: Physical Design Preparation (Medium Priority)

**Goal:** Prepare for ASIC fabrication using hierarchical/chiplet synthesis approach.

**Timeline:** 10-14 weeks (parallel tracks after FPGA validation)

**Strategy:** Bottom-up synthesis with DRC-clean hardened macros at each level.

```
Level 1 (Week 1-2):  Compute Primitives (~1mm² each)
Level 2 (Week 3-4):  Functional Blocks (~10-20mm² each)
Level 3 (Week 5-10): Subsystems (~50-400mm² each)
Level 4 (Week 11-14): Top Integration (800mm²)
```

#### 5A: Level 1 - Compute Primitives

| Block | Size | Reuse Count | Status |
|-------|------|-------------|--------|
| ternary_mac_array_64 | ~1mm² | ~2000× | 🟢 Config & RTL complete |
| rms_norm_block | ~0.5mm² | 60× | 🟢 Config & RTL complete |
| layer_norm_block | ~0.5mm² | 24× | 🟢 Config & RTL complete |
| softmax_unit | ~0.5mm² | 42× | 🟢 Config & RTL complete |
| silu_unit | ~0.3mm² | 42× | 🟢 Config & RTL complete |
| attention_head | ~2mm² | 414× | 🟢 Config & RTL complete |
| mlp_block | ~3mm² | 42× | 🟢 Config & RTL complete |

**Exit criteria:** Zero DRC violations, timing closed at 100MHz with 20% margin.

#### 5B: Level 2 - Functional Blocks

| Block | Size | Contains | Status |
|-------|------|----------|--------|
| transformer_block_llm | ~13mm² | 2×rms_norm, 9×attn_head, mlp | 🟢 Config & RTL complete |
| transformer_block_vision | ~20mm² | 2×layer_norm, 12×attn_head, mlp | 🟢 Config & RTL complete |
| projector_block | ~10mm² | 2-layer MLP (768→1152→576) | 🟢 Config & RTL complete |
| embedding_block | ~15mm² | Token + Position embeddings | 🟢 Config & RTL complete |

**Exit criteria:** Zero DRC, uses Level 1 macros, timing closed at 100MHz.

#### 5C: Level 3 - Subsystems

| Subsystem | Size | Contains | Status |
|-----------|------|----------|--------|
| vision_subsystem | ~250mm² | 12× transformer_block_vision + patch embed | 🟢 Config & RTL complete |
| llm_subsystem | ~400mm² | 30× transformer_block_llm + embedding | 🟢 Config & RTL complete |
| memory_subsystem | ~50mm² | DDR3 PHY + controller + AXI arbiter | 🟢 Config & RTL complete |
| io_subsystem | ~30mm² | Host IF, SPI, GPIO, Interrupt ctrl | 🟢 Config & RTL complete |

**Level 3 Files Created:**
- `openlane/level3/vision_subsystem/` - 12× vision transformers in 3×4 grid
- `openlane/level3/llm_subsystem/` - 30× LLM transformers with weight sharing
- `openlane/level3/memory_subsystem/` - 4-port AXI arbiter + DDR3 PHY
- `openlane/level3/io_subsystem/` - Parallel host IF + SPI + GPIO

**Exit criteria:** Zero DRC, uses Level 2 macros, timing closed at 100MHz.

**Next immediate steps:**
1. Create Level 4 top integration RTL (silens_soc_800mm.v)
2. Run OpenLane synthesis on Level 1 blocks (requires OpenLane installed)
3. Iterate on DRC at each level before moving up
4. Characterize timing/power for each hardened block

#### 5D: Level 4 - Top Integration

| Task | Description | Status |
|------|-------------|--------|
| 5D.1 | Create top-level RTL wrapper (silens_soc_top.v) | 🟢 Complete |
| 5D.2 | Create OpenLane config and floorplan | 🟢 Complete |
| 5D.3 | Create macro placement for all subsystems | 🟢 Complete |
| 5D.4 | Design power distribution network (25W) | 🔴 Needs OpenLane run |
| 5D.5 | Clock tree synthesis (100MHz, 28mm span) | 🔴 Needs OpenLane run |
| 5D.6 | Top-level routing | 🔴 Needs OpenLane run |
| 5D.7 | Final DRC/LVS signoff | 🔴 Needs OpenLane run |

**Level 4 Files Created:**
- `openlane/level4/silens_soc/config.json` - 800mm² synthesis config
- `openlane/level4/silens_soc/macro_placement.cfg` - Subsystem placement
- `openlane/level4/silens_soc/pin_order.cfg` - IO pin assignment
- `openlane/level4/silens_soc/src/silens_soc_top.v` - Top integration RTL
- `openlane/level4/silens_soc/src/silens_pll.v` - Clock generation
- `openlane/level4/silens_soc/src/silens_reset_sync.v` - Reset synchronizer

**OpenLane Files Created:**
- `openlane/level1/ternary_mac_array_64/` - MAC array synthesis
- `openlane/level1/rms_norm_block/` - RMS norm synthesis
- `openlane/level1/layer_norm_block/` - Layer norm synthesis (768-dim)
- `openlane/level1/softmax_unit/` - Softmax approximation
- `openlane/level1/silu_unit/` - SiLU activation
- `openlane/level1/attention_head/` - Single attention head
- `openlane/level1/mlp_block/` - SwiGLU MLP block
- `openlane/level2/transformer_block_llm/` - LLM transformer layer (13mm²)
- `openlane/level2/transformer_block_vision/` - Vision transformer layer (20mm²)
- `openlane/level2/projector_block/` - Vision-to-LLM projection (10mm²)
- `openlane/level2/embedding_block/` - Token/Position embeddings (15mm²)
- `openlane/level3/vision_subsystem/` - Vision encoder (250mm², 12× transformers)
- `openlane/level3/llm_subsystem/` - LLM decoder (400mm², 30× transformers)
- `openlane/level3/memory_subsystem/` - DDR3 + AXI arbiter (50mm²)
- `openlane/level3/io_subsystem/` - Host IF + peripherals (30mm²)
- `openlane/level4/silens_soc/` - **Top integration (800mm²)**
- `openlane/Makefile` - Hierarchical build orchestration
- `docs/architecture/HIERARCHICAL_SYNTHESIS_STRATEGY.md` - Full strategy doc

**Total RTL Blocks: 19 (7 Level 1 + 4 Level 2 + 4 Level 3 + 1 Level 4 + 3 support modules)**

---

### Phase 6: PCB Design (Lower Priority)

**Goal:** Design the physical board for the accelerator.

**Timeline:** 6-8 weeks (after FPGA validation)

| Task | Description | Status |
|------|-------------|--------|
| 6.1 | Schematic capture (power, PCIe, memory) | 🔴 Not started |
| 6.2 | Layer stackup for high-speed signals | 🔴 Not started |
| 6.3 | BGA breakout for chip package | 🔴 Not started |
| 6.4 | Thermal design (25W TDP) | 🔴 Not started |
| 6.5 | Signal integrity analysis | 🔴 Not started |
| 6.6 | Manufacturing file generation | 🔴 Not started |
| 6.7 | Order prototype PCBs | 🔴 Not started |

**PCB Specifications:**

| Parameter | Target |
|-----------|--------|
| Form Factor | Half-height PCIe card |
| Layers | 8-10 layer |
| Interface | PCIe 3.0 x4 |
| Power Input | 12V from PCIe slot |
| Cooling | Active (small fan) or passive heatsink |

---

### Phase 7: Crowdfunding Campaign (Parallel Track)

**Goal:** Secure funding for fabrication and team growth.

**Timeline:** Ongoing (launch when FPGA demo ready)

| Task | Description | Status |
|------|-------------|--------|
| 7.1 | Finalize Kickstarter campaign page | 🟡 Draft exists |
| 7.2 | Record video demo (FPGA prototype) | 🔴 Waiting for FPGA |
| 7.3 | Build pre-launch email list | 🔴 Not started |
| 7.4 | Set reward tiers and pricing | 🟡 Draft exists |
| 7.5 | Press kit and media outreach | 🟡 Draft exists |
| 7.6 | Launch campaign | 🔴 Not started |
| 7.7 | Backer updates during campaign | 🔴 Not started |

**Campaign Materials:**
- `docs/kickstarter/CAMPAIGN_PAGE.md` - Main campaign text
- `docs/kickstarter/REWARD_TIERS_BREAKDOWN.md` - Pricing structure
- `docs/kickstarter/PRESS_KIT.md` - Media materials
- `docs/kickstarter/FAQ_FULL.md` - Frequently asked questions

---

## Resource Requirements

### Hardware

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| Xilinx Kintex-7 FPGA board | Prototype platform | $500-2000 |
| PCIe development system | Host for FPGA | $1000-2000 |
| Logic analyzer | Debug | $300-500 |
| Oscilloscope | Signal analysis | $500-1000 |

### Software/Services

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| Vivado License | FPGA synthesis | Free (WebPACK) |
| OpenLane | ASIC synthesis | Free (open-source) |
| Cloud compute (GPU) | Model training/testing | $100-500/month |

### Fabrication (Future)

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| MPW shuttle slot | Prototype ASIC | $10,000-50,000 |
| Package and test | Assembly | $5,000-20,000 |
| PCB prototypes | Board fabrication | $500-2000 |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Timing closure failure | Medium | High | Start with lower clock, optimize |
| FPGA resource overflow | Medium | Medium | Reduce precision, optimize RTL |
| Model accuracy drop > 10% | Low | High | Mixed-precision quantization |
| PCIe integration issues | Medium | Medium | Start with USB fallback |
| Fabrication delays | Medium | Medium | FPGA as backup platform |
| Funding shortfall | Medium | High | Multiple funding sources |
| **SKY130 vocabulary mismatch** | **High** | **High** | **SPI primary (16-bit), parallel debug only** |
| **Parallel output power integrity** | Medium | Medium | Gray coding, 2x drive strength, stagger outputs |
| **Caravel pin conflicts** | Low | High | Validate against shuttle-specific docs |
| **SPI timing at 25 MHz** | Medium | Medium | Add SDC constraints, use 4x drive |

---

## Success Criteria

### Phase 1 (FPGA) - Must Have
- [ ] Inference runs on FPGA at > 10 images/second
- [ ] Latency < 100ms per image
- [ ] Power < 15W on FPGA
- [ ] Demo video showing live inference

### Phase 2 (SDK) - Must Have
- [ ] `pip install silens` works
- [ ] Example script processes image in < 5 lines of code
- [ ] Linux driver loads without errors

### Phase 3-4 (Quality) - Should Have
- [ ] Weight similarity > 90% with mixed precision
- [ ] VQA accuracy drop < 5%

### Phase 5-6 (ASIC) - Future Goal
- [ ] Confirm reticle size with SkyWater (critical path)
- [ ] Timing closure at 50-100 MHz (realistic for large die)
- [ ] Die area 350-400mm² monolithic OR 3× 250mm² MCM
- [ ] Power estimate 15-25W TDP
- [ ] Yield >10% (requires die size reduction from 800mm²)
- [ ] External DDR3 interface for KV cache
- [ ] BGA-900/FCBGA packaging

---

## Timeline Summary

```
2026 Q3 (Aug-Sep)
├── Phase 1: FPGA Prototyping ████████████████████
└── Phase 2: SDK Development  ████████████████

2026 Q4 (Oct-Dec)
├── Phase 3: Mixed Precision  ████████
├── Phase 4: Benchmarks       ████████████
├── Phase 5: Physical Design  ████████████████████
└── Phase 7: Kickstarter      ████████████████████████

2027 Q1 (Jan-Mar)
├── Phase 6: PCB Design       ████████████████████
└── Fabrication Prep          ████████████████████████
```

---

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Priority areas needing help:**
1. FPGA synthesis and optimization
2. Linux driver development
3. Python SDK implementation
4. PCB design review
5. Documentation and testing

---

## Contact

- **Repository:** https://github.com/loreii/SiLens
- **Issues:** https://github.com/loreii/SiLens/issues
- **Email:** hello@silens.ai

---

*This plan is a living document and will be updated as the project progresses.*
