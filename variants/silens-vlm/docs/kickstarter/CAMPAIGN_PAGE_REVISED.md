# SiLens DevKit — Open-Source Vision AI Development Platform

## The First Step Toward Hardwired Neural Networks

---

## What We're Building (And What We're NOT)

Let's be upfront:

**This campaign delivers the SiLens DevKit** — an FPGA-based development platform that runs our hardwired neural network architecture.

**This is NOT a custom ASIC.** That comes later.

We're doing this in two phases because building an ASIC from scratch costs $2.5-3M, and we believe in:
1. Proving the concept works before asking for millions
2. Building with our community, not just for them
3. Being honest about hardware development timelines

If you want to be part of the journey from FPGA to custom silicon, back us.

---

## The Vision: AI Etched in Silicon

Our end goal is revolutionary: **a chip where neural network weights are physical wire connections, not data in memory.**

That eliminates the memory bottleneck that limits all current AI accelerators.

The result (when we get to custom silicon):
- 100× faster than GPUs
- 1/5th the power
- Purpose-built for edge AI

**But we're not there yet.** This campaign funds the first step.

---

## What You Get: SiLens DevKit

### The Product

A PCIe card with:
- High-performance FPGA running SmolVLM-256M
- Same architecture as our future ASIC
- Same software interface (your code will work on both)
- Fully open-source design

### Specifications

| Spec | DevKit (This Campaign) | Future ASIC (2028+) |
|------|------------------------|---------------------|
| Model | SmolVLM-256M | SmolVLM-256M |
| Latency | 50-100ms | <5ms |
| Throughput | 5-10 img/sec | 200+ img/sec |
| Power | 40W | 25W |
| Interface | PCIe 3.0 x4 | PCIe 3.0 x4 |
| Price | $349-449 | $169-249 (target) |
| Software | Python SDK, Linux driver | Same |
| Open Source | Yes (RTL, PCB, software) | Yes |

### Why Back the DevKit?

1. **Develop today, deploy tomorrow** — Build your application now, upgrade to ASIC when it ships
2. **Shape the product** — Your feedback directly influences the ASIC design
3. **Priority access** — DevKit backers get first dibs on ASIC at discounted pricing
4. **Support open hardware** — Help prove there's a market for open AI accelerators

---

## Performance: Honest Numbers

### DevKit vs. Alternatives

| Platform | SmolVLM-256M Speed | Price | Power |
|----------|-------------------|-------|-------|
| CPU (i7-12700) | 0.5 img/sec | N/A | ~100W |
| SiLens DevKit | **5-10 img/sec** | **$349-449** | **40W** |
| RTX 4060 | 1-3 img/sec | $299 | 115W |
| RTX 4060 (batched) | 5-15 img/sec | $299 | 115W |

**Wait — the DevKit is similar to a GPU in speed?**

Yes. The FPGA version is constrained by FPGA limitations. The magic happens in the ASIC.

**So why buy the DevKit?**
- Same software, same interface — future-proof your code
- Dedicated hardware — no fighting for GPU resources
- Open source — modify and extend the design
- Community — help shape the final product
- Priority — first access to ASIC at $100 discount

---

## The Technology

### How Hardwired Neural Networks Work

Traditional approach:
```
Weights in memory → Load to compute → Calculate → Repeat
                        ↑
                    BOTTLENECK
```

Our approach:
```
Weights ARE the circuit → Compute instantly
```

Each weight becomes a wire:
- +1 = connected to power
- -1 = connected to ground
- Computation happens at electrical speed

### What the DevKit Proves

The DevKit implements this architecture on FPGA:
- Validates the RTL design
- Tests the software stack
- Proves the concept works
- Provides performance baseline

When we move to ASIC, we're not starting from scratch — we're hardening a proven design.

---

## What This Funds

### Budget Breakdown

| Category | Amount | % |
|----------|--------|---|
| FPGA modules & components | $65,000 | 32% |
| PCB design & prototyping | $35,000 | 17% |
| Engineering (RTL, firmware, drivers) | $40,000 | 20% |
| Assembly & production | $25,000 | 12% |
| Certification (FCC/CE) | $20,000 | 10% |
| Fulfillment & shipping | $15,000 | 8% |
| **Total** | **$200,000** | 100% |

### What This DOESN'T Fund

- ASIC tape-out (~$150K additional)
- ASIC wafer fabrication (~$100K additional)
- Full team salaries for 2+ years
- ASIC packaging and production

The DevKit is a stepping stone. The ASIC requires additional funding that we're pursuing through:
- Government grants (CHIPS Act programs)
- Seed investment (in parallel with this campaign)
- Future crowdfunding (if DevKit succeeds)

---

## Reward Tiers

### 🌱 SUPPORTER — $49
- Name in credits and README
- All development updates
- Discord access (Supporter role)
- $50 discount code for future ASIC

*Delivery: Immediate (digital)*

---

### 💻 DEVKIT EARLY BIRD — $349
*Limited to 200 backers*

- 1× SiLens DevKit (FPGA-based)
- Full software stack (drivers, SDK, examples)
- Printed quick-start guide
- Discord access (Developer role)
- $100 discount on future ASIC

*Delivery: October 2027*
*Retail value: $449*

---

### 💻 DEVKIT STANDARD — $449
- 1× SiLens DevKit (FPGA-based)
- Full software stack
- Printed documentation booklet
- Priority support queue
- Discord access (Developer role)
- $100 discount on future ASIC

*Delivery: October 2027*

---

### 🚀 DEVKIT + ASIC COMMITMENT — $599
- 1× SiLens DevKit (shipping Oct 2027)
- **Guaranteed ASIC unit at $149** (when available, ~2028-2029)
- All DevKit benefits
- Input on ASIC feature priorities

*You're pre-ordering both generations*
*Total value: $449 + $249 = $698*

---

### 👥 TEAM PACK — $1,299
- 3× SiLens DevKits
- Team onboarding call (30 min)
- Priority support
- 3× ASIC discount codes ($100 each)

*Delivery: October 2027*
*Per-unit: $433*

---

### 🏢 LAB PACK — $4,499
- 10× SiLens DevKits
- 2-hour integration consulting call
- Direct Slack channel with engineering
- 10× ASIC discount codes
- Logo on website

*Delivery: October 2027*
*Per-unit: $450*
*Limited to 10 backers*

---

## Stretch Goals

| Amount | Unlock |
|--------|--------|
| **$200,000** | ✓ Funded — DevKit ships! |
| **$300,000** | Windows driver support |
| **$400,000** | Second model variant (OCR-optimized) |
| **$500,000** | Begin ASIC design phase |
| **$750,000** | Accelerate ASIC — add second engineer |
| **$1,000,000** | ASIC tape-out funded! |

**The $1M stretch goal is ambitious but real** — if we hit it, the ASIC timeline accelerates by 6+ months.

---

## Timeline

| Milestone | Date |
|-----------|------|
| Campaign launch | September 2026 |
| Campaign ends | October 2026 |
| PCB design complete | January 2027 |
| Prototypes built | March 2027 |
| Certification | June 2027 |
| Production | August 2027 |
| **DevKit ships** | **October 2027** |

### ASIC Timeline (Requires Additional Funding)
| Milestone | Date |
|-----------|------|
| Design start | Q4 2027 |
| Tape-out | Q2 2028 |
| First silicon | Q4 2028 |
| **ASIC ships** | **Q1-Q2 2029** |

*ASIC dates are projections contingent on securing additional funding.*

---

## Risks and Challenges

### We're Being Honest

**This is hard.** Hardware development has real risks:

| Risk | Impact | Our Mitigation |
|------|--------|----------------|
| FPGA doesn't meet speed targets | 30-50% slower than spec | Performance range stated, not single number |
| PCB needs redesign | 2-3 month delay | Budget for 2 prototype iterations |
| Components unavailable | Cost increase, delay | Multiple suppliers identified |
| Certification fails first time | 1-2 month delay | Pre-compliance testing |
| We underestimated costs | Need to adjust scope | 20% contingency in budget |

**What we're NOT worried about:**
- The architecture works (validated in simulation)
- SmolVLM-256M is a real model (published by Hugging Face)
- We have FPGA experience (we've done this before)

### If Things Go Really Wrong

If we cannot deliver:
- Full transparency about what happened
- Refund of unspent funds
- All design work released open-source regardless

We'd rather fail openly than fail quietly.

---

## The Team

### Who We Are

[To be filled with actual team info]

**[Founder Name]** — Lead Architect
- [X] years in FPGA/ASIC design
- Previous: [Company]
- Why I'm doing this: [Personal motivation]

**[Engineer Name]** — Hardware Lead
- [Experience]

**Advisors:**
- [Advisor] — [Expertise]
- [Advisor] — [Expertise]

### What We've Built Before

- [Previous project with link]
- [Previous project with link]
- [Relevant open-source contribution]

---

## Why Open Source?

Everything about SiLens is open:
- RTL (Verilog) — Apache 2.0
- PCB schematics — Apache 2.0
- Drivers and SDK — Apache 2.0
- Documentation — CC-BY

**Why give it away?**

1. **Trust** — You can verify our claims
2. **Community** — Others can improve the design
3. **Longevity** — Project survives even if company doesn't
4. **Philosophy** — AI hardware should be accessible

---

## FAQ

### About the DevKit

**Q: Why should I buy this instead of just using my GPU?**
A: If you're happy with your GPU, stick with it! The DevKit is for people who want:
- Dedicated hardware that doesn't compete with other workloads
- Early access to our platform
- To support and shape open AI hardware
- Future-proof code that runs on our ASIC

**Q: Is 5-10 img/sec good?**
A: It's comparable to a GPU on this model. The DevKit proves the architecture; the ASIC delivers the speed.

**Q: Can I run other models?**
A: The DevKit runs SmolVLM-256M. The architecture could support other models but would require RTL modifications (it's open-source — you can do this!).

### About the ASIC

**Q: Will the ASIC actually happen?**
A: We're committed to it, but it requires additional funding beyond this campaign. Success here proves market demand, which makes raising that funding much easier.

**Q: When will the ASIC ship?**
A: Target is 2028-2029, but this depends on funding timing. DevKit backers will be first to know.

**Q: What if you never make the ASIC?**
A: You'll have a working DevKit, open-source designs, and the community we built together. Not nothing!

### About the Campaign

**Q: Why not just raise more money?**
A: We want to prove demand before asking for millions. $200K lets us deliver something real while building toward the bigger vision.

**Q: What if you don't hit $200K?**
A: Campaign fails, everyone gets refunded. We regroup and try again.

**Q: Why is the DevKit more expensive than the future ASIC?**
A: FPGAs are expensive! The ASIC will be cheaper because we're amortizing design costs over more units and custom silicon is more cost-effective at scale.

---

## Join Us

We're not promising to revolutionize AI hardware overnight.

We're promising to take the first step — openly, honestly, together.

If that resonates with you, back the SiLens DevKit.

**[BACK THIS PROJECT]**

---

*SiLens is a project of [Company Name]*
*SmolVLM-256M is developed by Hugging Face (Apache 2.0)*
*We are not affiliated with Hugging Face, SkyWater, or any FPGA vendor*
