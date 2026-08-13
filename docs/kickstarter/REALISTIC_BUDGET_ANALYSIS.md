# SiLens — Realistic Budget Analysis

## True Cost of Building a Custom ASIC from Scratch

---

## Reality Check

**$100,000 is nowhere near enough to build an ASIC product.**

This document provides an honest assessment of what it actually costs to:
1. Design a custom 800mm² ASIC
2. Build the PCB and supporting electronics
3. Prototype, fail, iterate, and finally ship a product

---

## Phase 1: ASIC Design & Verification

### RTL Design
| Item | Cost | Notes |
|------|------|-------|
| Senior ASIC Designer (contractor) | $180-250/hr | Need 2-3 for 12+ months |
| RTL design labor (12 months, 2.5 FTE) | **$600,000 - $900,000** | Core architecture, transformers, control |
| Verification engineer (12 months) | **$200,000 - $300,000** | UVM testbench, coverage |
| ML engineer (model optimization) | **$150,000 - $200,000** | Quantization, validation |

### EDA Tools
| Item | Cost | Notes |
|------|------|-------|
| OpenLane (open source) | $0 | But limited capabilities |
| Cadence/Synopsys (if needed) | $50,000 - $200,000/year | Startup programs available |
| Cloud compute for synthesis | $20,000 - $50,000 | Large design = long runs |
| FPGA prototyping boards | $10,000 - $30,000 | High-end FPGAs for validation |

### Design Consulting (External)
| Item | Cost | Notes |
|------|------|-------|
| Physical design consultant | $50,000 - $150,000 | Timing closure, P&R expertise |
| DFT consultant | $30,000 - $75,000 | Scan insertion, BIST |
| Signoff review | $20,000 - $50,000 | Pre-tape-out verification |

**Subtotal Phase 1: $1,130,000 - $1,955,000**

---

## Phase 2: Tape-Out & Fabrication

### SkyWater SKY130 Costs
| Item | Cost | Notes |
|------|------|-------|
| Mask set (full) | $100,000 - $150,000 | 800mm² is complex |
| Engineering lot (2-3 wafers) | $30,000 - $60,000 | First silicon for testing |
| Production wafers (initial) | $50,000 - $100,000 | Low volume to start |

### What If First Silicon Fails?
| Scenario | Additional Cost | Probability |
|----------|-----------------|-------------|
| Minor metal fix (ECO) | $30,000 - $50,000 | 30% |
| Major respin | $150,000 - $250,000 | 20% |
| Complete redesign | $300,000+ | 5% |

**Budget for 1 respin minimum: $150,000 - $250,000**

**Subtotal Phase 2: $330,000 - $560,000**

---

## Phase 3: PCB & Electronics Design

### PCB Design (THIS IS NOT TRIVIAL)
| Item | Cost | Notes |
|------|------|-------|
| PCB design engineer | $50,000 - $100,000 | 3-6 months, high-speed design |
| PCIe compliance expertise | $20,000 - $50,000 | External consultant |
| Signal integrity analysis | $15,000 - $30,000 | SI/PI simulation |
| 6+ layer PCB design | $10,000 - $25,000 | Gold fingers, impedance control |
| Schematic capture & review | $10,000 - $20,000 | Power delivery network |

### Power Management Design
| Item | Cost | Notes |
|------|------|-------|
| PMIC selection & design | $10,000 - $20,000 | 25W power delivery is non-trivial |
| Thermal analysis | $5,000 - $15,000 | Heatsink design |
| EMC pre-compliance | $10,000 - $25,000 | FCC/CE preparation |

### PCB Prototyping (Expect Failures!)
| Item | Cost | Notes |
|------|------|-------|
| Proto run #1 (10 boards) | $3,000 - $8,000 | Initial validation |
| Proto run #2 (fix issues) | $3,000 - $8,000 | 80% chance needed |
| Proto run #3 (final validation) | $3,000 - $8,000 | 40% chance needed |
| Component costs (3 proto runs) | $5,000 - $15,000 | BOM for testing |
| Assembly (3 proto runs) | $10,000 - $25,000 | PCBA at low volume |

**Budget for 2-3 PCB iterations: $25,000 - $60,000**

### Test Equipment
| Item | Cost | Notes |
|------|------|-------|
| Oscilloscope (high bandwidth) | $10,000 - $30,000 | PCIe debugging |
| Logic analyzer | $5,000 - $15,000 | Protocol analysis |
| Power supplies, multimeters | $3,000 - $8,000 | Lab basics |
| Thermal camera | $2,000 - $5,000 | Thermal debugging |
| PCIe test equipment | $5,000 - $20,000 | Compliance testing |

**Subtotal Phase 3: $161,000 - $394,000**

---

## Phase 4: Packaging & Assembly

### ASIC Packaging
| Item | Cost | Notes |
|------|------|-------|
| BGA package NRE | $15,000 - $40,000 | 625+ ball package tooling |
| Package development | $10,000 - $25,000 | Substrate, testing |
| Initial packaging run | $20,000 - $50,000 | First 100-200 units |

### Board Assembly
| Item | Cost | Notes |
|------|------|-------|
| CM setup/NRE | $5,000 - $15,000 | Stencils, fixtures, programming |
| Assembly (first 500 units) | $20,000 - $40,000 | ~$40-80/board at low volume |
| Testing & QC setup | $10,000 - $25,000 | Functional test development |

**Subtotal Phase 4: $80,000 - $195,000**

---

## Phase 5: Certification & Compliance

### Required Certifications
| Item | Cost | Notes |
|------|------|-------|
| FCC Part 15 (USA) | $10,000 - $25,000 | EMC testing + certification |
| CE marking (EU) | $10,000 - $20,000 | EMC + LVD |
| Pre-compliance testing | $5,000 - $15,000 | Before formal submission |
| PCIe compliance (optional) | $10,000 - $30,000 | For enterprise credibility |

### If You Fail EMC Testing
| Scenario | Additional Cost | Probability |
|----------|-----------------|-------------|
| Minor shielding fixes | $5,000 - $15,000 | 40% |
| PCB respin needed | $30,000 - $60,000 | 20% |

**Subtotal Phase 5: $35,000 - $90,000 (+ $30K contingency)**

---

## Phase 6: Software & Firmware

### Driver Development
| Item | Cost | Notes |
|------|------|-------|
| Linux kernel driver | $30,000 - $60,000 | PCIe driver, DMA |
| Windows driver (if promised) | $40,000 - $80,000 | Signed driver is complex |
| Firmware/microcode | $20,000 - $40,000 | If any embedded processor |

### SDK & Tools
| Item | Cost | Notes |
|------|------|-------|
| Python SDK | $15,000 - $30,000 | API, documentation |
| Demo applications | $10,000 - $20,000 | Showcase capabilities |
| Documentation | $10,000 - $20,000 | Technical docs, tutorials |

**Subtotal Phase 6: $85,000 - $250,000**

---

## Phase 7: Operations & Overhead

### Team Overhead (18-24 months)
| Item | Monthly | Total (20 months) |
|------|---------|-------------------|
| Office/lab space | $3,000 - $8,000 | $60,000 - $160,000 |
| Insurance (liability, D&O) | $1,000 - $3,000 | $20,000 - $60,000 |
| Legal (contracts, IP) | $2,000 - $5,000 | $40,000 - $100,000 |
| Accounting | $500 - $1,500 | $10,000 - $30,000 |
| Travel (vendors, conferences) | $1,000 - $3,000 | $20,000 - $60,000 |
| Misc (shipping, supplies) | $500 - $2,000 | $10,000 - $40,000 |

**Subtotal Phase 7: $160,000 - $450,000**

---

## Phase 8: Fulfillment & Shipping

### Production Run (1,000 units)
| Item | Cost | Notes |
|------|------|-------|
| ASIC (tested, packaged) | $80,000 - $120,000 | $80-120/unit at yield |
| PCB + assembly | $30,000 - $50,000 | $30-50/unit |
| Components | $15,000 - $25,000 | $15-25/unit |
| Packaging materials | $5,000 - $10,000 | Boxes, inserts |
| Shipping | $15,000 - $30,000 | Global fulfillment |

**Subtotal Phase 8: $145,000 - $235,000**

---

## TOTAL REALISTIC BUDGET

| Phase | Low Estimate | High Estimate |
|-------|--------------|---------------|
| 1. ASIC Design | $1,130,000 | $1,955,000 |
| 2. Tape-Out & Fab | $330,000 | $560,000 |
| 3. PCB & Electronics | $161,000 | $394,000 |
| 4. Packaging & Assembly | $80,000 | $195,000 |
| 5. Certification | $65,000 | $120,000 |
| 6. Software | $85,000 | $250,000 |
| 7. Overhead | $160,000 | $450,000 |
| 8. Fulfillment (1K units) | $145,000 | $235,000 |
| **Subtotal** | **$2,156,000** | **$4,159,000** |
| Contingency (25%) | $539,000 | $1,040,000 |
| **TOTAL** | **$2,695,000** | **$5,199,000** |

---

## Funding Strategy Options

### Option A: Kickstarter + Seed Round (Recommended)
| Source | Amount | Purpose |
|--------|--------|---------|
| Kickstarter | $300,000 - $500,000 | Market validation, working capital |
| Seed investment | $2,000,000 - $3,000,000 | Main development funding |
| Grants (CHIPS Act, etc.) | $100,000 - $500,000 | Supplement if available |
| **Total** | **$2,400,000 - $4,000,000** | |

**Kickstarter goal should be $300K-500K** — still aggressive but more honest about costs.

### Option B: Bootstrap with Smaller Scope
| Change | Savings | Trade-off |
|--------|---------|-----------|
| Smaller die (LLM only, no vision) | $300K - $500K | Less capable product |
| Longer timeline (founders do more) | $400K - $600K | 3+ years to ship |
| Skip Windows driver | $40K - $80K | Smaller market |
| Partner with university | $200K - $400K | Slower, IP complications |

### Option C: Two-Stage Campaign
1. **Stage 1: $150K** — Complete design, validate with FPGA
2. **Stage 2: $500K** — Tape-out and production (after Stage 1 proves feasibility)

This is more honest but harder to market.

---

## What $100,000 Actually Gets You

With only $100K, you can:
- [ ] NOT tape out an ASIC
- [ ] NOT produce a custom chip
- [x] Complete partial RTL design (maybe 30-40%)
- [x] Build an FPGA prototype
- [x] Validate the architecture concept
- [x] Create demos and documentation
- [x] Build community interest

**$100K is a "proof of concept" budget, not a "ship product" budget.**

---

## Honest Kickstarter Positioning

### What We Should Say

> "We're raising funds to complete the design phase and produce an FPGA-based development kit. Full ASIC production requires additional investment, which we're pursuing in parallel."

Or:

> "This Kickstarter is Phase 1 of our journey. Funds will be used for design and prototyping. We're also raising seed investment to cover the full $2.5M needed for ASIC production."

### Revised Funding Goal Options

| Goal | What It Covers | Honest Messaging |
|------|----------------|------------------|
| $150,000 | Design + FPGA prototype | "Pre-production validation" |
| $300,000 | Design + partial tape-out prep | "Design completion" |
| $500,000 | Design + first tape-out attempt | "Path to silicon" |
| $1,000,000+ | Realistic production budget | "Full production" |

---

## Recommendation

**Don't launch a $100K Kickstarter promising a finished ASIC product.**

Instead, consider:

1. **Raise seed funding first** ($2-3M) to de-risk the project
2. **Then launch Kickstarter** for community building and pre-orders
3. **Or be very transparent** that Kickstarter is partial funding

Many successful hardware Kickstarters (Pebble, Oculus) had VC backing behind them. The Kickstarter was for market validation and community, not primary funding.

---

## Risk of Underfunding

If you launch at $100K and succeed:
- You'll have money that can't complete the project
- You'll have obligations to backers
- You'll need to raise more money under pressure
- Failure to deliver destroys reputation

**It's better to not launch than to launch underfunded.**

---

*This analysis is meant to be sobering, not discouraging. Building hardware is expensive. Being honest about costs upfront builds trust and leads to better outcomes.*
