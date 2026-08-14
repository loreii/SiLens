# SiLens Development Plan

> **Last Updated:** August 14, 2026  
> **Status:** Active Development  
> **Version:** 0.1.0

This document outlines the development roadmap for the SiLens open-source hardwired vision-language AI accelerator.

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Architecture specification | 🟡 In progress | Core design complete |
| RTL design (Verilog) | 🟢 Complete | Core modules verified |
| RTL simulation (E2E pipeline) | 🟢 Complete | Icarus Verilog + cocotb |
| Model quantization tools | 🟢 Complete | Ternary quantization working |
| Semantic equivalence tests | 🟢 Complete | 94% visual, 87.9% weight similarity |
| FPGA prototype | 🔴 Not started | **Next priority** |
| Physical design (ASIC) | 🔴 Not started | Depends on FPGA validation |
| PCB design | 🔴 Not started | Depends on FPGA validation |
| SDK/Drivers | 🟡 In progress | Basic structure exists |
| Documentation | 🟢 Complete | E2E guides, test results |

### Key Achievements

- ✅ End-to-end RTL simulation pipeline verified
- ✅ Ternary quantization preserves 99.7% activation similarity
- ✅ Visual understanding validated with Lenna test (94% similarity)
- ✅ Comprehensive documentation for reproducibility

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

**Goal:** Prepare for ASIC fabrication with SkyWater SKY130 PDK.

**Timeline:** 4-6 weeks (after FPGA validation)

| Task | Description | Status |
|------|-------------|--------|
| 5.1 | Run OpenLane synthesis flow | 🔴 Not started |
| 5.2 | Analyze timing closure at target frequency | 🔴 Not started |
| 5.3 | Estimate die area | 🔴 Not started |
| 5.4 | Estimate power consumption | 🔴 Not started |
| 5.5 | Identify and optimize critical paths | 🔴 Not started |
| 5.6 | Generate preliminary GDSII | 🔴 Not started |
| 5.7 | DRC/LVS verification | 🔴 Not started |

**Target Specifications:**

| Parameter | Target | Notes |
|-----------|--------|-------|
| Process | SkyWater SKY130 | 130nm CMOS |
| Die Size | ~800mm² | Max reticle limit |
| Clock Frequency | 100-200 MHz | TBD based on timing |
| Power | 25W TDP | Active inference |
| Package | BGA-625 | Standard package |

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
- [ ] Timing closure at 100 MHz
- [ ] Die area < 800mm²
- [ ] Power estimate < 30W

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
