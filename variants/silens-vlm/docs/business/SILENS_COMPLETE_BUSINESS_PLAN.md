# SiLens™ Vision AI Accelerator
# Complete Business Plan

---

**Confidential - For Investor Review**  
**Version 1.0 | August 2026**

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Company Vision & Mission](#2-company-vision--mission)
3. [Problem Statement](#3-problem-statement)
4. [Solution Overview](#4-solution-overview)
5. [Technology & Product](#5-technology--product)
6. [Market Analysis](#6-market-analysis)
7. [Go-to-Market Strategy](#7-go-to-market-strategy)
8. [Operations](#8-operations)
9. [Team](#9-team)
10. [Financial Projections](#10-financial-projections)
11. [Funding Requirements](#11-funding-requirements)
12. [Risk Assessment](#12-risk-assessment)
13. [Exit Strategy](#13-exit-strategy)

---

# 1. Executive Summary

## The Opportunity

The AI hardware market is at an inflection point. While powerful vision-language models (VLMs) are transforming industries from healthcare to retail, deploying them at the edge remains prohibitively expensive, power-hungry, and locked behind proprietary ecosystems.

**We are building the world's first open-source hardwired multimodal vision-language AI accelerator.**

## What We're Building

SiLens is a PCIe 3.0 x4 accelerator card powered by a custom ASIC that implements SmolVLM-256M—a 246 million parameter multimodal AI model with its weights permanently etched in silicon.

| Key Specification | Value |
|-------------------|-------|
| Model | SmolVLM-256M (Vision + Language) |
| Parameters | 246 million |
| Process | SkyWater SKY130 (130nm, open PDK) |
| Die Size | ~800 mm² |
| Latency | <5ms end-to-end |
| Power | 25W |
| Price | $169-249 |
| License | Apache 2.0 (fully open) |

## Market Opportunity

- **TAM:** $52B AI accelerator market by 2027
- **SAM:** $8.5B edge inference accelerators
- **SOM:** $85M Year 3 (1% capture)

## Financial Highlights

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| Units Shipped | 0 | 3,000 | 35,000 |
| Revenue | $0 | $837K | $7.4M |
| Gross Margin | - | 61% | 57% |
| EBITDA | ($1.3M) | ($543K) | $2.3M |

## The Ask

**$2.5M Seed Round** to complete silicon design, tape-out, and initial production over 21 months.

---

# 2. Company Vision & Mission

## Our Vision

**A world where deploying capable AI is as accessible as deploying software.**

Just as Linux and open-source software democratized computing, open-source AI hardware will democratize intelligence at the edge.

## Our Mission

**To build the world's most accessible high-performance AI accelerator, fully open from transistors to training.**

## Core Values

| Value | Description |
|-------|-------------|
| **Radical Openness** | Apache 2.0 from RTL to manufacturing files |
| **Practical Impact** | Deployed systems solving real problems |
| **Accessible Excellence** | World-class engineering without world-class budgets |
| **Long-term Thinking** | Building infrastructure for a decade of edge AI |

## Five-Year North Star

- **100,000+ accelerators deployed** worldwide
- **Open designs** forked and improved by dozens of teams
- **Ecosystem of specialized models** for vertical applications
- **Proven thesis** that open-source AI hardware works

---

# 3. Problem Statement

## The Edge AI Deployment Gap

| Dimension | Problem |
|-----------|---------|
| **Cost** | RTX 4060: $299+ / H100: $25,000+ / Cloud APIs: $8,500/year |
| **Power** | GPUs need 100-300W; edge budgets are 10-50W |
| **Latency** | Real-time needs <10ms; GPUs deliver 20-100ms |
| **Memory** | Loading weights from memory dominates latency |

## Why Current Solutions Fail

| Solution | Limitation |
|----------|------------|
| **GPUs** | Too expensive, too power-hungry for edge |
| **Cloud APIs** | Too slow, too costly at scale |
| **Edge Accelerators** | Not multimodal (vision only) |
| **Custom ASICs** | $10-50M NRE, years of development |

## The Gap

**There is no affordable, low-power, low-latency solution for multimodal vision-language AI at the edge.**

---

# 4. Solution Overview

## SiLens: Hardwired Vision Intelligence

### The Core Innovation

Traditional accelerators store weights in memory. We **etch weights directly into silicon**.

```
Weight = +1  →  Metal trace to VDD
Weight = -1  →  Metal trace to GND
```

**Result:** Zero memory access, computation at electrical propagation speed.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SmolVLM-256M ASIC                        │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  VISION ENCODER: SigLIP-B/16 (93M params)             │ │
│  │  12 Transformer layers | 512×512 input → 64 tokens    │ │
│  └───────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  MULTIMODAL PROJECTOR (18M params)                    │ │
│  │  768-dim → 576-dim mapping                            │ │
│  └───────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  LANGUAGE MODEL: SmolLM2-135M (135M params)           │ │
│  │  30 Transformer layers | 49K vocab | 8K context       │ │
│  └───────────────────────────────────────────────────────┘ │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ PCIe x4 │  │  Power  │  │  Clock  │  │   I/O   │      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Performance Comparison

| Metric | SiLens | RTX 4060 | Advantage |
|--------|--------|----------|-----------|
| Latency | <5ms | 20-50ms | **5-10×** |
| Power | 25W | 115W | **4.6×** |
| Price | $169-249 | $299+ | **20-40%** |
| Images/sec | 1000+ | ~50 | **20×** |

### Capabilities

- ✅ Image captioning and description
- ✅ Visual question answering
- ✅ Document OCR and understanding
- ✅ Multi-image comparison
- ✅ Speculative decoding acceleration

---

# 5. Technology & Product

## 5.1 The Innovation: Hardwired Neural Networks

### Why 1-Bit Works

| Traditional Inference | 1-Bit Inference |
|----------------------|-----------------|
| Multiply-Accumulate (MAC) | XNOR + Popcount |
| 16-32 bit multipliers | Simple logic gates |
| Memory bandwidth limited | Compute-bound only |
| ~100-500 TOPS/W | ~10,000+ TOPS/W |

### Breaking the Memory Wall

- **Traditional:** Weights in VRAM → Memory Bus → Cache → Compute (microseconds)
- **Hardwired:** Weights ARE the circuit → Direct computation (nanoseconds)

## 5.2 Product Specifications

### ASIC Specifications

| Parameter | Specification |
|-----------|---------------|
| Model | SmolVLM-256M |
| Total Parameters | 246 million |
| Vision Encoder | SigLIP-B/16 (93M) |
| Language Model | SmolLM2-135M (135M) |
| Process | SkyWater SKY130 (130nm) |
| Die Size | ~800 mm² |
| Core Voltage | 1.8V |
| Clock | 100-200 MHz |
| Package | BGA-625 |

### PCIe Card Specifications

| Parameter | Specification |
|-----------|---------------|
| Interface | PCIe 3.0 x4 (4 GB/s) |
| Form Factor | Half-height, half-length |
| Power | 25W TDP (slot powered) |
| Cooling | Passive heatsink |
| Dimensions | 168mm × 69mm |

## 5.3 Technology Roadmap

| Phase | Timeline | Product | Process |
|-------|----------|---------|---------|
| 1 | Now-Month 21 | SmolVLM-256M | SKY130 (130nm) |
| 2 | Month 24-36 | SmolVLM-500M | 65nm |
| 3 | Month 36-48 | SmolVLM-2B | Multi-die |

## 5.4 Intellectual Property

### Open Foundation (Apache 2.0)

- SmolVLM model weights
- SkyWater SKY130 PDK
- OpenLane RTL-to-GDSII tools

### Proprietary Differentiation

- Physical design optimizations
- Weight-to-routing algorithms
- System integration IP
- Manufacturing know-how

### Inherent IP Protection

Weights physically encoded in silicon—extraction requires destructive analysis.

---

# 6. Market Analysis

## 6.1 Market Size

| Segment | 2024 | 2027 | CAGR |
|---------|------|------|------|
| AI Accelerators (TAM) | $28B | $52B | 23% |
| Edge AI (SAM) | $8B | $18B | 31% |
| Multimodal Inference | $2B | $8.5B | 33% |

## 6.2 Target Customer Segments

| Segment | % Revenue | Pain Point | Volume |
|---------|-----------|------------|--------|
| **AI Startups** | 35% | GPU costs, latency | 5-20 units |
| **Edge Deployment** | 30% | Power, privacy | 50-500 units |
| **Research** | 15% | Budget, access | 10-50 units |
| **Hobbyists** | 10% | Cost, local inference | 1-3 units |
| **Enterprise** | 10% | Scale, efficiency | 100-1000+ units |

## 6.3 Competitive Analysis

| Factor | SiLens | NVIDIA | Google Coral | Intel Movidius |
|--------|--------|--------|--------------|----------------|
| Price | $169-249 | $300-4K+ | $100-150 | $80-150 |
| Power | 25W | 50-300W | 2-4W | 1-2W |
| Latency | <5ms | 10-50ms | 15-30ms | 30-100ms |
| Multimodal | ✅ | ⚠️ | ❌ | ❌ |
| Open Source | ✅ | ❌ | ⚠️ | ❌ |

## 6.4 Positioning

> **For AI developers and edge deployers** who need fast, affordable multimodal inference, **SiLens** is the **open-source AI accelerator** that delivers **sub-5ms vision-language processing at 25W**.

---

# 7. Go-to-Market Strategy

## 7.1 Phase 1: Community Launch (Months 1-6)

**Objective:** Build developer mindshare and validate product-market fit

| Tactic | Details |
|--------|---------|
| Crowdfunding | Launch on Crowd Supply / Kickstarter for pre-orders |
| Open Source | Release all hardware designs on GitHub |
| Influencer Seeding | Send units to 50 key ML YouTubers, researchers, OSS maintainers |
| Benchmarks | Publish comprehensive comparisons vs. competition |
| Community | Launch Discord for early adopters |

**Success Metrics:**
- 1,000 pre-orders
- 500 GitHub stars
- 10 community benchmark/review videos
- 1,000 Discord members

**Target Revenue:** $250K

## 7.2 Phase 2: Developer Traction (Months 6-12)

**Objective:** Establish as go-to solution for multimodal inference

| Tactic | Details |
|--------|---------|
| Model Zoo | Partner with Hugging Face on optimized model conversions |
| Framework Integration | Partner with ONNX, TensorRT-lite alternatives |
| Case Studies | Publish 10 early adopter success stories |
| Startup Program | Free units for YC/Techstars companies |

**Success Metrics:**
- 5,000 units sold
- 25 optimized models in Model Zoo
- 50 startup partnerships
- 10 published case studies

**Target Revenue:** $1M

## 7.3 Phase 3: Enterprise Expansion (Months 12-24)

**Objective:** Capture edge deployment and enterprise segments

| Tactic | Details |
|--------|---------|
| Enterprise SKU | Launch with support SLA |
| Channel Partners | Partner with systems integrators |
| Events | Embedded World, Edge AI Summit presence |
| Direct Sales | Build team of 2-3 enterprise AEs |
| Reference Architectures | Vertical-specific solutions |

**Success Metrics:**
- 3 systems integrator partnerships
- 5 enterprise pilot deployments
- 25,000 cumulative units sold
- 2 Fortune 500 customers

**Target Revenue:** $5M

## 7.4 Phase 4: Scale (Months 24-36)

**Objective:** Achieve market leadership in accessible multimodal inference

| Tactic | Details |
|--------|---------|
| Gen 2 Product | Launch second-generation (65nm) |
| Geographic Expansion | EU, APAC distribution |
| Channel Program | Build partner ecosystem |
| OEM Partnerships | Strategic design wins |

**Target Revenue:** $15M+

## 7.5 Pricing Strategy

| SKU | Price | Target Segment | Margin |
|-----|-------|----------------|--------|
| SiLens Core | $169 | Hobbyists, makers | 45% |
| SiLens Pro | $249 | Startups, edge deployment | 55% |
| SiLens Enterprise | $349 | Enterprise (with support) | 60% |
| Volume (100+) | Custom | Edge fleet, enterprise | 50%+ |

**Academic Discount:** 25% off for verified institutions

## 7.6 Strategic Partnerships

### Technology Partners

| Partner Type | Target Partners | Value Exchange |
|--------------|-----------------|----------------|
| Model Providers | Hugging Face, Replicate | Optimized models → distribution |
| ML Frameworks | ONNX Runtime, Apache TVM | Integration → ecosystem access |
| Edge Platforms | Balena, Edge Impulse | Hardware reference → customer access |

### Distribution Partners

| Partner Type | Target Partners | Focus |
|--------------|-----------------|-------|
| Developer Distributors | Adafruit, SparkFun, Seeed | Hobbyist/maker segment |
| Electronics Distributors | Digi-Key, Mouser, Arrow | Broad availability |
| Systems Integrators | Regional edge AI integrators | Enterprise deployment |

---

# 8. Operations

## 8.1 Development Timeline

**Total Timeline:** 21 months from project initiation to production-ready silicon

```
Month:  1   3   6   9   12  15  18  21
        ├───┼───┼───┼───┼───┼───┼───┤
Phase 1 ████░░░░░░░░░░░░░░░░░░░░░░░░  Validation (3 mo)
Phase 2 ░░░░████████░░░░░░░░░░░░░░░░  Scaled Prototype (6 mo)
Phase 3 ░░░░░░░░░░░░████████░░░░░░░░  Full Integration (6 mo)
Phase 4 ░░░░░░░░░░░░░░░░░░░░████░░░░  Tape-out (3 mo)
Phase 5 ░░░░░░░░░░░░░░░░░░░░░░░░████  Bring-up (3 mo)
```

### Phase 1: Architecture Validation (Months 1-3)
- Implement 4-layer toy vision encoder and LLM in Verilog
- Verify against PyTorch references
- Build Python→Verilog weight generator pipeline
- Go/no-go decision point

### Phase 2: Scaled Prototype (Months 4-9)
- Generate full RTL for SigLIP-B/16 (93M params)
- Generate full RTL for SmolLM2-135M (135M params)
- Complete OpenLane synthesis on all blocks
- Power analysis and thermal modeling

### Phase 3: Full Integration (Months 10-15)
- Integrate PCIe Gen3 x4 controller IP
- Top-level SoC integration
- DRC/LVS verification
- FPGA prototype validation

### Phase 4: Tape-out (Months 16-18)
- Final timing closure
- Parasitic extraction
- Generate production GDSII
- Submit to SkyWater Technology

### Phase 5: Bring-up (Months 19-21)
- Receive engineering samples
- PCIe link bring-up
- Full model inference validation
- Linux driver development

## 8.2 Manufacturing Strategy

### Fabrication Partner: SkyWater Technology

| Parameter | Specification |
|-----------|---------------|
| Process | SKY130 (130nm CMOS) |
| Metal Layers | 5 |
| Operating Voltage | 1.8V core, 3.3V I/O |
| Max Reticle Size | ~800mm² usable |
| Cycle Time | ~12 weeks from tape-out |

### Production Scaling

| Phase | Volume | Timeline | Unit Cost |
|-------|--------|----------|-----------|
| Engineering Samples | 50-100 | Month 19-20 | NRE |
| Pilot Production | 500 | Month 21-22 | $140-160 |
| Initial Production | 1,000-2,000 | Month 23-24 | $120-140 |
| Volume Production | 5,000+ | Month 25+ | $90-110 |

### Yield Management

At 800mm² die size with 0.5-1.0 defects/cm², expected yield: **30-50%**

**Yield Improvement Strategies:**
- Redundancy circuits in embedding tables
- Post-silicon voltage/frequency binning
- Foundry collaboration on process optimization

## 8.3 Supply Chain

### Bill of Materials (Target at 5,000 units)

| Component | Est. Unit Cost |
|-----------|---------------|
| SmolVLM-256M ASIC (tested) | $60-80 |
| BGA Package | $5-8 |
| PCB (6-layer, gold fingers) | $8-12 |
| PMIC + regulators | $3-5 |
| Clock generator | $1-2 |
| Passives | $2-3 |
| Heatsink + thermal pad | $3-5 |
| PCIe bracket | $2-3 |
| Assembly + test | $8-12 |
| **Total BOM** | **$92-130** |

---

# 9. Team

## 9.1 Organizational Structure

```
                        ┌─────────────────────┐
                        │     CEO/Founder     │
                        └──────────┬──────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
┌─────────▼─────────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
│ VP of Engineering │   │   VP of Operations │   │   VP of Business  │
└─────────┬─────────┘   └─────────┬─────────┘   └───────────────────┘
          │                       │
    ┌─────┴─────┐           ┌─────┴─────┐
    │           │           │           │
┌───▼───┐ ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
│ RTL   │ │ Physical  │ │ Supply│ │ Quality   │
│ Design│ │ Design    │ │ Chain │ │ Assurance │
└───────┘ └───────────┘ └───────┘ └───────────┘
```

## 9.2 Team Composition by Phase

### Phase 1-2: Design & Development (Months 1-9)

| Role | Headcount |
|------|-----------|
| Chief Architect | 1 |
| RTL Design Engineers | 3-4 |
| Verification Engineers | 2 |
| ML Engineer | 1 |
| Physical Design Lead | 1 |
| **Total** | **8-9** |

### Phase 3-4: Integration & Tape-out (Months 10-18)

| Role | Additional Headcount |
|------|---------------------|
| Physical Design Engineers | 2-3 |
| DFT Engineer | 1 |
| Power Integrity Engineer | 1 |
| Firmware Engineer | 1 |
| **Running Total** | **13-15** |

### Phase 5+: Production & Scale (Months 19-21+)

| Role | Additional Headcount |
|------|---------------------|
| Test Engineer | 1 |
| Supply Chain Manager | 1 |
| Quality Engineer | 1 |
| Applications Engineer | 1 |
| **Steady State** | **17-19** |

## 9.3 Key Hires (Critical Positions)

### Immediate (Pre-funding)

| Position | Priority | Rationale |
|----------|----------|-----------|
| Chief Architect | P0 | Define architecture before design |
| Lead RTL Engineer | P0 | Own transformer implementation |
| ML/Quantization Engineer | P0 | Ensure model→hardware quality |

### Series A Hires (Months 1-6)

| Position | Priority | Rationale |
|----------|----------|-----------|
| Physical Design Lead | P1 | Critical for 800mm² die |
| Verification Lead | P1 | Prevent costly re-spins |
| DFT Engineer | P2 | Testability for yield |

---

# 10. Financial Projections

## 10.1 Revenue Forecast

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| **Units Sold** | 0 | 3,000 | 35,000 |
| **ASP** | - | $229 | $189 |
| **Hardware Revenue** | $0 | $687K | $6.6M |
| **Support/Services** | $0 | $50K | $300K |
| **Licensing Revenue** | $0 | $100K | $500K |
| **Total Revenue** | **$0** | **$837K** | **$7.4M** |

## 10.2 Profitability Path

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| **Revenue** | $0 | $837K | $7.4M |
| **COGS** | $0 | $330K | $3.2M |
| **Gross Profit** | $0 | $507K | $4.2M |
| **Gross Margin** | - | 61% | 57% |
| **Operating Expenses** | $1.275M | $1.05M | $1.9M |
| **EBITDA** | ($1.275M) | ($543K) | $2.3M |
| **EBITDA Margin** | - | (65%) | 31% |

## 10.3 Unit Economics

### Cost Structure by Volume

| Component | 1K Units | 5K Units | 10K Units |
|-----------|----------|----------|-----------|
| ASIC die | $80 | $65 | $50 |
| BGA Package | $10 | $8 | $6 |
| PCB + Assembly | $25 | $22 | $18 |
| Other Components | $15 | $12 | $11 |
| Test & QA | $5 | $4 | $3 |
| **Total COGS** | **$135** | **$111** | **$88** |

### Margin Analysis

| Volume | COGS | ASP | Gross Margin % |
|--------|------|-----|----------------|
| Launch (1K) | $135 | $249 | 46% |
| Growth (5K) | $111 | $199 | 44% |
| Scale (10K) | $88 | $169 | 48% |
| Volume (25K+) | $75 | $159 | 53% |

## 10.4 Breakeven Analysis

| Metric | Value |
|--------|-------|
| Fixed Costs (NRE) | $300,000 |
| Contribution Margin @ Launch | $114/unit |
| Unit Breakeven | 2,632 units |
| **Time to Breakeven** | **Month 28 (Y2 Q4)** |
| **Cash Flow Breakeven** | **Month 30 (Y3 Q2)** |

---

# 11. Funding Requirements

## 11.1 The Ask: $2.5M Seed Round

| Tranche | Amount | Purpose | Timeline |
|---------|--------|---------|----------|
| Tranche 1 | $1.0M | Design & verification | Months 1-9 |
| Tranche 2 | $1.0M | Tape-out & prototyping | Months 10-18 |
| Tranche 3 | $0.5M | Production ramp & working capital | Months 19-24 |

## 11.2 Use of Funds

### Phase 1: Design & Development (Months 1-12) — $1.2M

| Category | Allocation |
|----------|------------|
| Engineering Team | $600K |
| EDA Tools & Licenses | $50K |
| Design Verification | $100K |
| IP Licensing | $75K |
| Cloud Compute | $50K |
| Facilities & Equipment | $75K |
| Legal & IP | $50K |
| Travel & Conferences | $25K |
| Contingency | $175K |

### Phase 2: Tape-Out & Bring-Up (Months 13-18) — $900K

| Category | Allocation |
|----------|------------|
| Mask Set (SKY130) | $100K |
| Wafer Fabrication | $150K |
| Packaging | $50K |
| PCB Prototyping | $25K |
| Test Development | $75K |
| Engineering Team | $375K |
| Bring-up Equipment | $50K |
| Contingency | $75K |

### Phase 3: Production Prep (Months 19-24) — $400K

| Category | Allocation |
|----------|------------|
| First Production Run | $150K |
| Manufacturing Setup | $50K |
| Quality & Compliance | $40K |
| Sales & Marketing | $100K |
| Working Capital | $60K |

## 11.3 Proposed Terms

| Term | Value |
|------|-------|
| Instrument | SAFE or Priced Seed |
| Raise Amount | $2.5M |
| Valuation Cap | $7.5M |
| Discount | 20% |
| Pro-rata Rights | Yes |
| Board Seat | 1 (if >$500K) |

## 11.4 Cap Table (Post-Seed)

| Shareholder | % Ownership |
|-------------|-------------|
| Founders | 60% |
| Employee Pool | 15% |
| Seed Investors | 25% |

---

# 12. Risk Assessment

## 12.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Die size exceeds target** | Medium (30-40%) | High | 20% area margin; fallback to smaller model |
| **Timing closure failure** | Medium (25-35%) | High | Conservative 100MHz target; multi-clock design |
| **Silicon functionality failure** | Low-Med (15-25%) | Critical | Extensive verification; FPGA prototype; re-spin budget |
| **Model quality degradation** | Low (15-20%) | Medium | Early RTL vs. PyTorch validation |

## 12.2 Manufacturing Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Low yield at 800mm²** | Medium (30-40%) | High | Assume 30% worst-case in pricing; redundancy circuits |
| **Foundry capacity constraints** | Low-Med (20-30%) | Medium | Early capacity planning; secondary foundry qualification |
| **OSAT/assembly delays** | Medium (25-35%) | Medium | Qualify multiple partners |
| **Component shortage** | Low (15-20%) | Low-Med | No single-source components; safety stock |

## 12.3 Business Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Market timing** | Medium (30-40%) | Medium | Focus on unique value prop; close customer connection |
| **Competitive response** | Med-High (40-50%) | Medium | Price leadership; community moat; first-mover advantage |
| **Funding gap** | Medium (30%) | High | Milestone-based development; strategic investors; grants |

## 12.4 Execution Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Key personnel departure** | Medium (25-35%) | High | Competitive comp; no single points of failure; documentation |
| **Schedule slip** | High (50-60%) | Medium | 20% schedule buffer; aggressive risk management |

## 12.5 Contingency Plans

### Technical Fallbacks

| Trigger | Contingency |
|---------|-------------|
| Die area >850mm² | Reduce to language model only |
| Timing fails at 100MHz | Accept 50MHz (still meets latency) |
| First silicon dead | Execute re-spin (budget reserved) |
| Yield <25% | Transition to MCM architecture |

### Business Fallbacks

| Trigger | Contingency |
|---------|-------------|
| Series A not closed by Month 6 | Reduce team; extend Phase 1-2 |
| Major competitor launches | Pivot to software/IP licensing |
| Manufacturing partner fails | Port design to GlobalFoundries 130nm |

---

# 13. Exit Strategy

## 13.1 Exit Scenarios

### Scenario A: Strategic Acquisition (Most Likely)

| Timeline | Acquirer Type | Multiple |
|----------|---------------|----------|
| Year 4-5 | Semiconductor company | 5-8× revenue |
| Year 4-5 | Cloud/AI platform | 8-12× revenue |
| Year 5-7 | System integrator | 4-6× revenue |

**Target Acquirers:**
- NVIDIA, AMD, Intel (GPU/accelerator consolidation)
- Qualcomm, MediaTek (edge AI expansion)
- AWS, Google, Microsoft (custom silicon programs)
- Lattice, Microchip (specialty accelerator play)

### Scenario B: Growth Equity / Series B (Year 3-4)

| Metric | Value |
|--------|-------|
| Y3 Revenue | $7.4M |
| Growth Rate | >200% |
| Projected Y4 Revenue | $18M |
| Series B Raise | $15-25M |
| Pre-money Valuation | $50-75M |

### Scenario C: IPO Track (Year 6-7)

Requires $50M+ ARR with path to $100M+. Possible if edge AI market expands dramatically.

## 13.2 Return Analysis for Seed Investors

### Base Case: $50M Acquisition (Year 5)

| Metric | Value |
|--------|-------|
| Seed Investment | $2.5M |
| Seed Ownership | 25% |
| Exit Valuation | $50M |
| Seed Proceeds | $12.5M |
| **Return Multiple** | **5.0×** |
| **IRR** | **38%** |

### Upside Case: $100M Acquisition (Year 5)

| Metric | Value |
|--------|-------|
| Seed Ownership (post-dilution) | 20% |
| Exit Valuation | $100M |
| Seed Proceeds | $20M |
| **Return Multiple** | **8.0×** |
| **IRR** | **52%** |

### Downside Case: $20M Acquisition (Year 4)

| Metric | Value |
|--------|-------|
| Seed Ownership | 25% |
| Exit Valuation | $20M |
| Seed Proceeds | $5M |
| **Return Multiple** | **2.0×** |
| **IRR** | **19%** |

## 13.3 Comparable Transactions

| Company | Acquirer | Year | Value | Multiple |
|---------|----------|------|-------|----------|
| Habana Labs | Intel | 2019 | $2B | >20× (strategic) |
| Nervana | Intel | 2016 | $400M | Pre-revenue |
| DeePhi Tech | Xilinx | 2018 | $300M | >15× |
| Annapurna Labs | Amazon | 2015 | $350M | N/A |

---

# Appendix A: NRE Budget Summary

| Category | Low | High | Budgeted |
|----------|-----|------|----------|
| EDA tools & licenses | $20K | $50K | $35K |
| IP licensing | $0 | $30K | $15K |
| FPGA prototyping | $10K | $20K | $15K |
| Computing infrastructure | $15K | $30K | $22K |
| SkyWater mask set | $80K | $120K | $100K |
| Engineering wafer lot | $20K | $40K | $30K |
| Package development | $10K | $20K | $15K |
| PCB design & prototypes | $10K | $20K | $15K |
| Assembly setup | $5K | $10K | $8K |
| Test program development | $15K | $25K | $20K |
| Certification (FCC, CE) | $30K | $50K | $40K |
| Contingency (20%) | $43K | $83K | $63K |
| **Total NRE** | **$258K** | **$498K** | **$378K** |

---

# Appendix B: Contact Information

**SiLens Technologies, Inc.**

*For investor inquiries and additional information*

---

**Document Version:** 1.0  
**Last Updated:** August 2026  
**Classification:** Confidential - For Investor Review

---

*This document contains forward-looking statements and projections based on current assumptions. Actual results may vary. Investment involves risk.*
