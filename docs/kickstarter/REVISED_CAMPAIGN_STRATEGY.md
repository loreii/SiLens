# SiLens — Revised Campaign Strategy

## Realistic Approach to Hardware Crowdfunding

---

## The Problem with the Original Plan

| Issue | Reality |
|-------|---------|
| $100K goal | Covers ~4% of actual costs |
| "Ship product in 18 months" | Unrealistic for ASIC from scratch |
| No external consulting budget | PCB/electronics design needs experts |
| No respin budget | First silicon often fails |
| No prototype iteration budget | PCB will need 2-3 revisions |

---

## Three Viable Strategies

### Strategy A: FPGA Development Kit First (Recommended)

**Concept:** Ship an FPGA-based product that demonstrates the architecture, then use revenue to fund ASIC development.

| Phase | Timeline | Deliverable | Cost |
|-------|----------|-------------|------|
| 1 | 6 months | FPGA dev kit | $150K |
| 2 | 18 months | ASIC (with seed funding) | $2.5M |

**Kickstarter Campaign:**
- **Goal:** $150,000
- **Product:** SiLens DevKit (FPGA-based)
- **Price:** $299-399
- **Delivery:** 8-10 months
- **Honest pitch:** "This is Step 1 toward custom silicon"

**Pros:**
- Achievable with Kickstarter alone
- Proves market demand
- Generates revenue for ASIC R&D
- Lower risk for backers
- Faster delivery builds trust

**Cons:**
- FPGA won't match ASIC performance
- Higher unit cost
- Less dramatic pitch

---

### Strategy B: Seed Round + Kickstarter (Pre-Orders)

**Concept:** Raise proper funding first, use Kickstarter for community and pre-orders.

| Source | Amount | Use |
|--------|--------|-----|
| Seed round | $2.5M | Development & production |
| Kickstarter | $300-500K | Working capital, community |

**Kickstarter Campaign:**
- **Goal:** $300,000
- **Product:** SiLens ASIC card
- **Price:** $199-299 (can afford to price aggressively with VC backing)
- **Delivery:** 18-24 months
- **Honest pitch:** "We're VC-backed, Kickstarter is for the community"

**Pros:**
- Properly funded
- Can deliver on promises
- Community input on product
- Price can be more competitive

**Cons:**
- Need VC first (chicken-egg)
- VC may not want crowdfunding
- Less "underdog" appeal

---

### Strategy C: Transparent Two-Phase Campaign

**Concept:** Be completely honest that this is a multi-stage project.

**Phase 1 Kickstarter:**
- **Goal:** $200,000
- **Deliverable:** Detailed design, FPGA validation, early access program
- **Price:** $99-149 (design access + future discount)
- **Delivery:** 12 months (design package, not hardware)

**Phase 2 (later):**
- Crowdfunding or seed round for production
- Phase 1 backers get priority pricing

**Pros:**
- Completely honest
- Lower risk for backers
- Builds genuine community
- Validates interest before big commitment

**Cons:**
- Hard to market "pay for design, not product"
- Backers may not understand
- Two campaigns is harder

---

## Recommended: Strategy A (FPGA DevKit)

### Product: SiLens DevKit

**What It Is:**
- PCIe card with high-end FPGA (Xilinx Kintex/Artix Ultrascale)
- SmolVLM-256M implemented in RTL
- Same software interface as future ASIC
- Open-source design files

**Specs:**
| Metric | FPGA DevKit | Future ASIC |
|--------|-------------|-------------|
| Latency | ~50-100ms | <5ms |
| Throughput | 10-20 img/sec | 200+ img/sec |
| Power | 35-50W | 25W |
| Price | $349-449 | $169-249 |

**Honest Positioning:**
> "SiLens DevKit is your path to developing on our platform today. It runs the same model, same software, at 10-20× GPU speed. When our custom ASIC ships, you'll have a seamless upgrade path — and DevKit backers get priority pricing."

### Revised Budget for FPGA DevKit

| Category | Cost |
|----------|------|
| **FPGA & Components** | |
| FPGA module (Kintex-7/Artix-7 US) | $200-400/unit |
| PCB design (simpler than ASIC) | $20,000-40,000 |
| PCB prototyping (2-3 iterations) | $15,000-30,000 |
| Production PCB (500 units) | $15,000-25,000 |
| Components per unit | $50-80 |
| Assembly per unit | $30-50 |
| **Engineering** | |
| RTL optimization for FPGA | $50,000-80,000 |
| PCB design consulting | $20,000-40,000 |
| Firmware/drivers | $30,000-50,000 |
| **Operations** | |
| Certification (FCC/CE) | $20,000-40,000 |
| Fulfillment | $20,000-40,000 |
| Overhead (12 months) | $50,000-100,000 |
| **Contingency (20%)** | $50,000-100,000 |
| **TOTAL** | **$300,000-575,000** |

**Unit Economics (500 units):**
| Item | Cost |
|------|------|
| FPGA module | $300 |
| PCB + assembly | $80 |
| Components | $65 |
| Packaging/shipping | $25 |
| **COGS** | **$470** |
| **Price** | **$599 (early bird) / $699 (retail)** |
| **Margin** | **$129-229 (22-33%)** |

Wait — that's expensive. Let's reconsider...

---

### Alternative: Smaller FPGA, Lower Price

**Use a smaller FPGA (Artix-7 35T or similar):**
- Can still run SmolVLM-256M at reduced speed
- FPGA module cost: $100-150
- Target price: $399 early bird

| Item | Cost |
|------|------|
| FPGA module | $120 |
| PCB + assembly | $60 |
| Components | $50 |
| Packaging/shipping | $20 |
| **COGS** | **$250** |
| **Price** | **$399 (early bird) / $499 (retail)** |
| **Margin** | **$149-249 (37-50%)** |

**Performance tradeoff:**
- ~5-10 images/second (still 5× faster than GPU in some cases)
- Proof of concept, not production performance

---

## Revised Kickstarter Campaign: FPGA DevKit

### Campaign Summary

| Parameter | Value |
|-----------|-------|
| **Product** | SiLens DevKit (FPGA-based) |
| **Goal** | $200,000 |
| **Duration** | 30 days |
| **Early Bird Price** | $349 (limited 200) |
| **Standard Price** | $449 |
| **Delivery** | 10-12 months |

### Reward Tiers

| Tier | Price | Units | What You Get |
|------|-------|-------|--------------|
| **Supporter** | $49 | 0 | Updates, Discord, name in credits |
| **DevKit Early Bird** | $349 | 1 | FPGA DevKit + all software | Limited 200 |
| **DevKit Standard** | $449 | 1 | FPGA DevKit + all software |
| **DevKit + ASIC Priority** | $549 | 1 | DevKit now + $100 off future ASIC |
| **Team Pack** | $1,299 | 3 | 3× DevKits + priority support |
| **Lab Pack** | $3,999 | 10 | 10× DevKits + consulting hour |

### Stretch Goals

| Amount | Unlock |
|--------|--------|
| $200K | Funded ✓ |
| $300K | Windows driver support |
| $400K | Additional model (OCR-optimized variant) |
| $500K | Begin ASIC design phase |
| $750K | Accelerate ASIC timeline |

### Key Messaging Changes

**OLD (Unrealistic):**
> "We're building a custom ASIC that's 100× faster than a GPU"

**NEW (Honest):**
> "SiLens DevKit lets you develop on our platform today. It's 5-10× faster than CPU inference, with the same software interface as our future ASIC. Back us to prove the market exists — and get first access when we go to silicon."

---

## Campaign Page Revisions Needed

### Section: What You're Backing

**Add this section:**

> ### What You're Getting: The DevKit
> 
> This Kickstarter delivers the **SiLens DevKit** — an FPGA-based development platform running our hardwired neural network architecture.
>
> **This is NOT the final ASIC product.** It's the development version that lets you:
> - Start building applications today
> - Validate your use case
> - Provide feedback that shapes the final product
> - Get priority access when the ASIC ships
>
> | | DevKit (This Campaign) | Future ASIC |
> |---|---|---|
> | Availability | 10-12 months | 24-30 months |
> | Performance | 5-10 img/sec | 200+ img/sec |
> | Power | 40W | 25W |
> | Price | $349-449 | $169-249 |
> | Software | Same | Same |

### Section: The Road to Silicon

> ### Our Two-Phase Plan
>
> **Phase 1: DevKit (This Campaign)**
> - Prove the architecture works
> - Build developer community
> - Generate revenue for Phase 2
> - Deliver: Q3 2027
>
> **Phase 2: Custom ASIC (Future)**
> - Requires additional funding (~$2.5M)
> - We're pursuing grants and investment in parallel
> - DevKit backers get priority pricing
> - Target delivery: 2028-2029
>
> We're being transparent: the ASIC is the destination, but the DevKit is the journey. Back us if you want to be part of both.

---

## Updated Financial Reality

### FPGA DevKit Campaign

**If we raise $200K and sell 500 units:**

| Revenue | |
|---------|---|
| 500 units @ $400 average | $200,000 |
| Kickstarter fees (5%) | -$10,000 |
| Payment processing (3%) | -$6,000 |
| **Net Revenue** | **$184,000** |

| Costs | |
|-------|---|
| COGS (500 × $250) | $125,000 |
| Engineering | $30,000 |
| Certification | $15,000 |
| Fulfillment | $15,000 |
| **Total Costs** | **$185,000** |

| **Net** | **-$1,000** |

Basically break-even — which is fine for a Kickstarter. The value is:
1. Proving market demand
2. Building community
3. Generating code/IP for ASIC phase
4. Credibility for seed round

### Path to ASIC

With a successful DevKit campaign:
- Proven market (500+ customers waiting for ASIC)
- Working RTL (validated on FPGA)
- Community and developer ecosystem
- Revenue to bootstrap seed round prep

This makes raising $2.5M seed MUCH easier.

---

## Summary: What Changes

| Original Plan | Revised Plan |
|---------------|--------------|
| $100K goal | $200K goal |
| Ship ASIC in 18 months | Ship FPGA DevKit in 10-12 months |
| $149-249 price | $349-449 price |
| 100× faster than GPU | 5-10× faster than GPU (for now) |
| Promise everything | Honest about what this phase delivers |

**The ASIC vision remains the same.** We're just being honest about the path to get there.

---

*"Under-promise and over-deliver" beats "over-promise and fail to deliver" every time.*
