# SiLens™ — The World's First Open-Source Vision AI Accelerator

## 100× Faster Than a GPU. Half the Price. 25 Watts.

---

![SiLens Hero Image Placeholder]

---

## TL;DR

**SiLens is a PCIe card that runs vision + language AI at 200 images/second with <5ms latency — using just 25 watts.**

We achieve this by doing something no one else has done: **etching the AI model weights directly into silicon.** No memory. No bottlenecks. Just raw, instant inference.

- 🚀 **100× faster** than a $300 GPU running the same model
- ⚡ **25W power** vs. 115W for a graphics card
- 💰 **$149-249** — cheaper than the GPU it replaces
- 🔓 **Fully open-source** — Apache 2.0 from silicon to software
- 🤖 **Multimodal AI** — sees images AND understands language

**Back us to bring democratized AI hardware to the world.**

---

## The Problem: AI Hardware is Broken

You want to run AI locally. Maybe you're building a smart camera system, processing documents, or just want to chat with images without sending them to the cloud.

Your options today:

| Option | Price | Power | Latency | Problem |
|--------|-------|-------|---------|---------|
| **Cloud API** | $0.01/image | N/A | 500ms+ | Privacy, ongoing costs, internet required |
| **Consumer GPU** | $300+ | 115W | 300ms+ | Overkill, power-hungry, needs a PC |
| **Edge TPU** | $75-150 | 2-4W | 30ms | Vision only, no language understanding |
| **Enterprise AI** | $10,000+ | 300W+ | <10ms | Absurdly expensive |

**There's nothing in between.** Nothing that's affordable, efficient, AND capable of understanding both images and text.

Until now.

---

## The Solution: AI Baked Into Silicon

### How Traditional AI Works

```
[Image] → [Load weights from memory] → [Compute] → [Answer]
                    ↑
            This is the bottleneck
```

A GPU has to load billions of weight values from memory for every single inference. Memory bandwidth — not compute power — is what limits speed.

### How SiLens Works

```
[Image] → [Weights ARE the circuit] → [Answer]
                    ↑
            No memory access needed
```

We encode each model weight as a physical wire connection:
- **Weight = +1** → Wire to power (VDD)
- **Weight = -1** → Wire to ground (GND)

**The model IS the chip.** Computation happens at the speed of electricity moving through wires — nanoseconds, not milliseconds.

---

## What Can SiLens Do?

SiLens runs **SmolVLM-256M**, a state-of-the-art vision-language model with 246 million parameters. It can:

### 📸 Describe Images
*"What's in this photo?"*
> "A golden retriever playing fetch on a sandy beach at sunset. The dog is mid-leap, catching a red frisbee."

### ❓ Answer Questions About Images
*"How many people are in this room?"*
> "There are 7 people visible — 4 seated at the conference table and 3 standing near the whiteboard."

### 📄 Read and Understand Documents
*"Extract the total from this receipt"*
> "The total is $47.83, including $3.42 tax."

### 🔍 Compare Multiple Images
*"What changed between these two photos?"*
> "The red car in the parking lot has moved, and a person with a blue umbrella has appeared near the entrance."

### ⚡ Process Video in Real-Time
At 200+ images/second, SiLens can analyze live video feeds for:
- Security and surveillance
- Quality control in manufacturing
- Traffic monitoring
- Retail analytics

---

## Performance: The Numbers

### SiLens vs. $300 GPU (RTX 4060)

| Metric | RTX 4060 | **SiLens** | **Improvement** |
|--------|----------|------------|-----------------|
| Price | $299 | $149-249 | **20-50% cheaper** |
| Single-image latency | 300-1000ms | **<5ms** | **60-200× faster** |
| Throughput (single) | 1-3 img/sec | **200+ img/sec** | **100× faster** |
| Throughput (batch) | 5-15 img/sec | **1000+ img/sec** | **70× faster** |
| Power consumption | 115W | **25W** | **4.6× efficient** |
| Form factor | Full desktop GPU | **Half-height PCIe** | Fits anywhere |

### Why Such a Huge Difference?

The RTX 4060 is a **general-purpose GPU** with 8GB of VRAM, 18 TFLOPS of compute, and support for thousands of different operations. It's incredibly flexible.

But SmolVLM-256M only needs **500MB** of that memory. It uses a **tiny fraction** of that compute power. You're paying for capabilities you'll never use.

SiLens is **purpose-built**. Every transistor is dedicated to running one model as fast as physically possible. No wasted silicon. No wasted power. No wasted money.

---

## Technical Specifications

### The ASIC

| Specification | Value |
|---------------|-------|
| Model | SmolVLM-256M |
| Total Parameters | 246 million |
| Vision Encoder | SigLIP-B/16 (93M parameters) |
| Language Model | SmolLM2-135M (135M parameters) |
| Process Node | SkyWater SKY130 (130nm) |
| Die Size | ~800mm² |
| Core Voltage | 1.8V |
| Clock Frequency | 100-200 MHz |

### The Card

| Specification | Value |
|---------------|-------|
| Interface | PCIe 3.0 x4 |
| Form Factor | Half-height, half-length |
| Dimensions | 168mm × 69mm |
| Power | 25W TDP (slot-powered) |
| Cooling | Passive heatsink |
| Weight | ~150g |

### Software Support

| Platform | Status |
|----------|--------|
| Linux | Full support (kernel driver + Python API) |
| Windows | Community support planned |
| macOS | Not supported (no PCIe) |
| Docker | Official container images |
| Python API | `pip install silens` |
| C/C++ API | Native library included |
| ONNX Runtime | Integration planned |

---

## Reward Tiers

### 🌱 SEED — $49
**The Believer**
- Name in credits (README + website)
- Digital thank-you card
- Early access to development updates
- Discord role: Founding Supporter

*Estimated delivery: Immediate*

---

### 🔧 MAKER — $149
**SiLens Core** (Early Bird — Limited to 500)
- 1× SiLens PCIe accelerator card
- Low-profile bracket included
- Quick-start guide
- 1 year of software updates
- Discord role: Early Adopter

*Estimated delivery: March 2028*
*Retail price: $169*

---

### ⚡ DEVELOPER — $199
**SiLens Pro** (Early Bird — Limited to 300)
- 1× SiLens PCIe accelerator card
- Full-height AND low-profile brackets
- Premium heatsink option
- Printed technical documentation
- Priority support queue
- Discord role: Developer

*Estimated delivery: March 2028*
*Retail price: $249*

---

### 🏢 STUDIO — $499
**SiLens Pro × 3**
- 3× SiLens PCIe accelerator cards
- All Pro tier benefits
- Multi-card setup guide
- 30-minute onboarding call
- Discord role: Studio

*Estimated delivery: March 2028*
*Retail price: $747*

---

### 🏭 ENTERPRISE — $1,499
**SiLens Pro × 10**
- 10× SiLens PCIe accelerator cards
- Extended 2-year support
- Custom integration consultation (2 hours)
- Priority bug fixes
- Logo on website (Sponsors section)
- Discord role: Enterprise Partner

*Estimated delivery: March 2028*
*Retail price: $2,490*

---

### 🚀 PIONEER — $4,999
**Founding Partner Package**
- 25× SiLens PCIe accelerator cards
- Lifetime software updates
- Direct Slack channel with team
- Quarterly roadmap calls
- Logo prominently featured
- First access to Gen 2 hardware
- Discord role: Pioneer

*Estimated delivery: March 2028*
*Limited to 10 backers*

---

### 💎 VISIONARY — $14,999
**Strategic Partner**
- 100× SiLens PCIe accelerator cards
- All Pioneer benefits
- Custom model optimization consultation
- On-site deployment support (if needed)
- Board observer rights (optional)
- Named in press releases

*Estimated delivery: March 2028*
*Limited to 3 backers*

---

## Stretch Goals

### $100,000 — FUNDED ✓
Base goal achieved! Production begins.

### $250,000 — Enhanced Heatsink
Upgraded passive cooling solution for sustained workloads.

### $500,000 — Windows Driver
Official Windows support with signed drivers.

### $750,000 — Model Zoo
10 additional optimized models (OCR specialist, object detection, etc.)

### $1,000,000 — Gen 1.5 Development
Begin development of improved 65nm version with 2× performance.

### $1,500,000 — RISC-V Co-processor
Add a small RISC-V core for custom pre/post-processing.

---

## Timeline

| Milestone | Date |
|-----------|------|
| **Campaign Launch** | September 2026 |
| **Campaign Ends** | October 2026 |
| **Design Finalization** | December 2026 |
| **Tape-out to Foundry** | June 2027 |
| **First Silicon** | September 2027 |
| **Validation Complete** | December 2027 |
| **Mass Production** | January 2028 |
| **Shipping Begins** | March 2028 |

---

## The Team

### [Founder Name] — CEO & Chief Architect
[Bio placeholder — ASIC design experience, previous projects]

### [Name] — VP Engineering
[Bio placeholder — physical design, large-die experience]

### [Name] — ML Lead
[Bio placeholder — model optimization, quantization]

### Advisors
- [Advisor 1] — [Expertise]
- [Advisor 2] — [Expertise]

---

## Why Open Source?

We believe AI hardware should be as open as AI software.

**Everything about SiLens is open:**
- RTL design files (Verilog)
- PCB schematics and layout
- BOM and assembly instructions
- Linux kernel driver
- Python SDK
- Documentation

**License:** Apache 2.0 — use it for anything, including commercial projects.

**Why give it away?**

1. **Trust** — You can verify exactly what the hardware does
2. **Community** — Others can improve our designs
3. **Longevity** — Even if we disappear, the project lives on
4. **Ecosystem** — More users = more models = more value for everyone

---

## Risks and Challenges

We believe in transparency. Here's what could go wrong:

### Manufacturing Delays
**Risk:** Semiconductor manufacturing is complex. Unexpected issues could delay tape-out or production.
**Mitigation:** We've built 6 months of buffer into our timeline. We have relationships with multiple packaging partners.

### Yield Issues
**Risk:** At 800mm² die size, manufacturing yield may be lower than expected, increasing costs.
**Mitigation:** We've priced conservatively assuming 30% yield. Actual yield may be better (40-50%), which would reduce costs.

### Performance Variance
**Risk:** First silicon may not hit exact performance targets.
**Mitigation:** Our architecture is robust to clock speed variations. Even at 50% of target speed, we still beat GPUs by 50×.

### Supply Chain
**Risk:** Component shortages could delay PCB assembly.
**Mitigation:** We use only commodity components with multiple suppliers. No exotic parts.

### Software Maturity
**Risk:** Drivers and SDK may have bugs at launch.
**Mitigation:** Open-source development means the community can help find and fix issues quickly.

---

## Frequently Asked Questions

### Is this vaporware?

No. We have:
- Working RTL simulation of the core architecture
- Verified bit-accurate match with PyTorch reference
- Preliminary synthesis results from OpenLane
- Relationships with SkyWater Technology (foundry)

This campaign funds the tape-out and production run.

### Why 130nm? Isn't that ancient?

Yes, 130nm is old by modern standards. But:
1. **It's open** — SkyWater SKY130 is the only fully open PDK
2. **It's cheap** — Mask costs are ~$100K vs. $10M+ for modern nodes
3. **It's enough** — Our architecture doesn't need bleeding-edge transistors

Future generations will move to 65nm and beyond as open PDKs mature.

### Can I run other models?

Not on Gen 1. The weights are physically etched into silicon — you can't change them.

However, we're exploring:
- Mask-programmable variants (change model via new mask set)
- Future architectures with limited weight flexibility

### What about training?

SiLens is inference-only. You cannot train models on it. For training, use GPUs or cloud services, then deploy the trained model to SiLens.

### Does it work with my existing AI software?

We provide:
- Python API compatible with common workflows
- ONNX Runtime integration (planned)
- Drop-in replacement for GPU inference in many cases

You'll need to adapt your code slightly, but we provide migration guides.

### What if the campaign doesn't reach its goal?

If we don't hit $100,000, all backers are fully refunded. No partial funding.

### Can I get a refund after backing?

Yes, until the campaign ends. After that, standard Kickstarter refund policies apply.

---

## Press & Media

For press inquiries, review units, or interviews:
**press@silens.ai**

Press kit available at: **silens.ai/press**

---

## Join the Revolution

The era of accessible AI hardware begins now.

For too long, running AI locally meant choosing between:
- Expensive, power-hungry GPUs
- Weak, limited edge devices
- Privacy-compromising cloud APIs

**SiLens changes everything.**

Back us today and be part of building the future of open AI hardware.

---

**[BACK THIS PROJECT]**

---

*SiLens™ is a trademark of SiLens Technologies, Inc.*
*SmolVLM is developed by Hugging Face and used under Apache 2.0 license.*
*SkyWater SKY130 PDK is provided by Google and SkyWater Technology under Apache 2.0 license.*
