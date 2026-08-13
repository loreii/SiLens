# SiLens Kickstarter — Risks and Challenges

## Comprehensive Risk Disclosure for Campaign Page

---

## Our Commitment to Transparency

Hardware crowdfunding is inherently risky. We believe backers deserve complete honesty about what could go wrong. This document outlines every significant risk we've identified and what we're doing to mitigate them.

**Bottom line:** We're experienced engineers who've designed chips before. We've built contingencies into our timeline and budget. But semiconductors are hard, and delays happen. Back us because you believe in the vision and understand the risks.

---

## 1. Manufacturing Risks

### Risk: Silicon Doesn't Work on First Try

**What could happen:** First silicon arrives and doesn't function correctly due to design bugs, manufacturing defects, or integration issues.

**Probability:** 15-25%

**Impact:** 4-6 month delay, +$100-150K cost for respin

**Our mitigation:**
- Extensive pre-silicon verification (>95% coverage target)
- FPGA prototyping before tape-out
- Conservative design margins
- Budget and timeline include one potential respin

**What it means for backers:** Possible 4-6 month delay from estimated delivery date.

---

### Risk: Low Manufacturing Yield

**What could happen:** At 800mm² die size, yield (percentage of working chips) may be lower than expected, increasing per-unit costs.

**Probability:** 30-40%

**Impact:** Higher COGS, potential price increase for post-campaign sales

**Our mitigation:**
- Priced assuming 30% yield (conservative)
- Redundancy circuits in critical areas
- Working with SkyWater on process optimization
- If yield is better (40-50%), costs decrease

**What it means for backers:** Campaign pricing is locked. Risk is on us, not you.

---

### Risk: Foundry Capacity or Delays

**What could happen:** SkyWater Technology experiences capacity constraints, equipment issues, or other delays that push out our fabrication schedule.

**Probability:** 20-30%

**Impact:** 2-4 month delay

**Our mitigation:**
- Early engagement with SkyWater on capacity planning
- Design is portable to other 130nm foundries if needed
- Buffer built into timeline

**What it means for backers:** Possible delay; we'll communicate transparently.

---

### Risk: Packaging Partner Issues

**What could happen:** Our OSAT (outsourced semiconductor assembly and test) partner has quality issues or delays with the BGA packaging.

**Probability:** 20-30%

**Impact:** 2-4 month delay

**Our mitigation:**
- Qualifying multiple OSAT partners (primary + backup)
- Using standard package types with multiple suppliers
- Starting qualification early (before silicon arrives)

**What it means for backers:** Possible delay; backup suppliers reduce risk.

---

## 2. Technical Risks

### Risk: Performance Below Target

**What could happen:** Silicon works but doesn't hit our target specifications (clock speed, latency, power).

**Probability:** 20-30%

**Impact:** Reduced performance claims, potentially lower perceived value

**Our mitigation:**
- Conservative design targets (100MHz vs. potential 200MHz)
- Architecture is robust to frequency variations
- Even at 50% of target speed, we still beat GPUs by 50×

**What it means for backers:** Worst case, you get a product that's "only" 50× faster than a GPU instead of 100×. Still game-changing.

---

### Risk: Model Quality Degradation

**What could happen:** The hardwired 1-bit model produces worse outputs than expected compared to the full-precision reference.

**Probability:** 15-20%

**Impact:** Reduced accuracy on some tasks

**Our mitigation:**
- Bit-accurate RTL simulation vs. PyTorch reference
- Using validated quantization techniques (BitNet b1.58)
- Higher precision (8-bit) activations preserve quality
- Extensive testing before tape-out

**What it means for backers:** Model quality will be validated before manufacturing. No surprises.

---

### Risk: Thermal Issues

**What could happen:** The chip runs hotter than expected, requiring active cooling or throttling.

**Probability:** 15-20%

**Impact:** Redesigned heatsink, potential performance throttling

**Our mitigation:**
- Detailed thermal modeling before tape-out
- Conservative 25W power budget
- Heatsink designed with margin
- Pro tier includes enhanced cooling option

**What it means for backers:** Worst case, we upgrade the heatsink (at our cost) or provide a small fan attachment.

---

## 3. Supply Chain Risks

### Risk: Component Shortages

**What could happen:** Critical components (PMICs, connectors, PCBs) become unavailable or have extended lead times.

**Probability:** 15-20%

**Impact:** 1-3 month delay, potential cost increase

**Our mitigation:**
- No single-source components in BOM
- Alternate parts qualified during design
- 3-6 month safety stock for long-lead items
- All components are commodity items (no exotic parts)

**What it means for backers:** Minor risk; we've designed for resilience.

---

### Risk: PCB/Assembly Partner Issues

**What could happen:** Contract manufacturer has quality or capacity issues.

**Probability:** 15-20%

**Impact:** 1-2 month delay

**Our mitigation:**
- Using established CM partners with hardware crowdfunding experience
- Small pilot run before full production
- Inspection and testing protocols

**What it means for backers:** Unlikely to cause major delays.

---

## 4. Financial Risks

### Risk: Campaign Underfunds

**What could happen:** We don't reach our $100,000 goal.

**Probability:** Unknown (depends on market reception)

**Impact:** Project doesn't proceed

**Our mitigation:**
- Extensive pre-launch marketing
- Building community before launch
- Realistic goal based on minimum viable production

**What it means for backers:** Full refund if goal not met. Kickstarter standard.

---

### Risk: Costs Exceed Budget

**What could happen:** Unexpected expenses push us over budget, threatening project completion.

**Probability:** 25-30%

**Impact:** Need to raise additional funds or reduce scope

**Our mitigation:**
- 15-20% contingency built into budget
- Phased development with go/no-go checkpoints
- Stretch goals provide additional buffer
- Founder investment available as bridge if needed

**What it means for backers:** We're committed to delivering. We'll find a way.

---

### Risk: Currency Fluctuations

**What could happen:** USD weakens against currencies in our supply chain (Asia), increasing costs.

**Probability:** 30-40% (some fluctuation likely)

**Impact:** Margin compression

**Our mitigation:**
- Priced with margin for currency movement
- US-based foundry (SkyWater) for major expense
- Can adjust non-essential features if needed

**What it means for backers:** Risk is on us.

---

## 5. Timeline Risks

### Risk: Overall Schedule Slip

**What could happen:** Accumulation of small delays pushes delivery beyond our estimate.

**Probability:** 50-60% (some delay is likely)

**Impact:** 2-6 months later than estimated

**Our mitigation:**
- 6 months of buffer built into timeline
- Aggressive milestone tracking
- Parallel workstreams where possible
- Regular backer updates on progress

**What it means for backers:** Hardware is hard. Expect March 2028, but June-September 2028 is possible if things go wrong.

---

### Risk: Certification Delays

**What could happen:** FCC/CE certification takes longer than expected or requires design changes.

**Probability:** 20-30%

**Impact:** 1-3 month delay

**Our mitigation:**
- Engaging certification consultants early
- Design for compliance from the start
- Pre-compliance testing before formal submission

**What it means for backers:** Minor risk; we're planning for this.

---

## 6. External Risks

### Risk: Competitor Launches Similar Product

**What could happen:** A larger company (NVIDIA, Google, etc.) launches a competing edge AI product.

**Probability:** 25-35%

**Impact:** Reduced differentiation, pricing pressure

**Our mitigation:**
- First-mover advantage in hardwired VLM space
- Open-source creates community moat
- Price leadership ($149-249 vs. enterprise products)
- Focus on specific use cases where we excel

**What it means for backers:** Even if competitors emerge, you'll have a unique open-source product at a great price.

---

### Risk: Regulatory Changes

**What could happen:** New export controls, tariffs, or regulations affect our ability to manufacture or ship.

**Probability:** 10-20%

**Impact:** Varies (could be minor to significant)

**Our mitigation:**
- US-based foundry (SkyWater) avoids most export concerns
- Monitoring regulatory developments
- Legal counsel on compliance

**What it means for backers:** Low probability; we're staying informed.

---

### Risk: Economic Downturn

**What could happen:** Recession reduces demand for AI hardware, affecting our ability to scale post-campaign.

**Probability:** 20-30%

**Impact:** Slower growth, but campaign backers still fulfilled

**Our mitigation:**
- Campaign funds are segregated for backer fulfillment
- Lean operations allow us to weather downturns
- AI hardware remains high-priority even in downturns

**What it means for backers:** Your rewards are funded by your pledge. Economic conditions affect our growth, not your delivery.

---

## 7. Execution Risks

### Risk: Key Team Member Leaves

**What could happen:** Critical engineer or leader departs mid-project.

**Probability:** 20-30%

**Impact:** 2-4 month delay, knowledge loss

**Our mitigation:**
- Competitive compensation with meaningful equity
- No single point of failure (cross-training)
- Strong documentation practices
- Advisor network can help fill gaps

**What it means for backers:** We're building a resilient team, not a single-founder dependency.

---

### Risk: Design Tool Issues

**What could happen:** Open-source EDA tools (OpenLane, etc.) have bugs or limitations we didn't anticipate.

**Probability:** 20-30%

**Impact:** Design iterations, 1-2 month delay

**Our mitigation:**
- Deep familiarity with OpenLane toolchain
- Contributing fixes back to community
- Commercial tool licenses available as backup

**What it means for backers:** We know these tools well. Minor risk.

---

## 8. What We're NOT Worried About

To be clear, some things are NOT significant risks:

### ✅ Technical Feasibility
The architecture is proven. 1-bit neural networks work. Hardwired weights work. The question is execution, not feasibility.

### ✅ Model Performance
SmolVLM-256M is a published, validated model from Hugging Face. Its capabilities are known.

### ✅ Market Demand
Edge AI is growing 30%+ annually. The need for fast, affordable inference is clear.

### ✅ Open-Source Commitment
We're philosophically committed to open source. This isn't a marketing gimmick.

---

## Our Promise to Backers

1. **Transparency**: Monthly updates, good news and bad
2. **Communication**: We'll tell you about problems before they become crises
3. **Commitment**: We'll exhaust every option before considering cancellation
4. **Honesty**: If we can't deliver, you'll know why

---

## Risk Summary Matrix

| Risk | Probability | Impact | Mitigation Quality |
|------|-------------|--------|-------------------|
| Silicon respin needed | 15-25% | High | Strong |
| Low yield | 30-40% | Medium | Strong |
| Foundry delays | 20-30% | Medium | Good |
| Performance below target | 20-30% | Low-Medium | Strong |
| Component shortage | 15-20% | Low | Strong |
| Schedule slip | 50-60% | Medium | Good |
| Cost overrun | 25-30% | Medium | Good |
| Competitor launch | 25-35% | Low-Medium | Good |
| Key person leaves | 20-30% | Medium | Good |

---

## Final Thoughts

Every hardware project has risks. We've been honest about ours because we respect your investment — both financial and emotional.

We're not asking you to take a leap of faith. We're asking you to make an informed decision to support a team that's done the homework, built the contingencies, and is committed to delivering.

If you're comfortable with the risks outlined here, we'd be honored to have you as a backer.

If not, we understand. Follow our progress, and maybe join us for Gen 2.

---

*Risk disclosure version: 1.0*
*Last updated: August 2026*
