# SiLens Kickstarter — Comprehensive FAQ

## For Campaign Page, Backer Support, and Press Inquiries

---

## Table of Contents

1. [Product & Technology](#1-product--technology)
2. [Performance & Comparisons](#2-performance--comparisons)
3. [Compatibility & Software](#3-compatibility--software)
4. [Campaign & Rewards](#4-campaign--rewards)
5. [Shipping & Fulfillment](#5-shipping--fulfillment)
6. [Company & Team](#6-company--team)
7. [Technical Deep Dive](#7-technical-deep-dive)
8. [Future Plans](#8-future-plans)

---

## 1. Product & Technology

### What is SiLens?

SiLens is a PCIe accelerator card that runs a vision-language AI model (SmolVLM-256M) with the model weights physically etched into the silicon. This eliminates memory access bottlenecks and enables extremely fast, low-power inference.

### What does "hardwired" mean?

Traditional AI accelerators store model weights in memory (RAM/VRAM) and load them for each computation. In SiLens, each weight value is encoded as a physical wire connection in the chip:
- Weight = +1 → Wire connected to VDD (power)
- Weight = -1 → Wire connected to GND (ground)

This means the model IS the circuit — no memory access required.

### What model does SiLens run?

SiLens runs **SmolVLM-256M**, a 246-million parameter vision-language model developed by Hugging Face. It consists of:
- **SigLIP-B/16**: A vision encoder with 93M parameters
- **SmolLM2-135M**: A language model with 135M parameters
- **Multimodal projector**: 18M parameters connecting vision to language

### Can I run different models on SiLens?

No. The model weights are physically etched into the silicon and cannot be changed. SiLens is purpose-built for SmolVLM-256M.

However:
- Future products may support different models
- We're exploring mask-programmable variants for custom models
- The open-source design can be modified to create chips with other models

### Is SiLens a GPU?

No. SiLens is an **inference-only application-specific integrated circuit (ASIC)**. Unlike a GPU:
- It cannot run arbitrary programs
- It cannot train models
- It runs only the hardwired model
- It has no general-purpose memory

Think of it as a "model in a chip" rather than a programmable accelerator.

### What can SmolVLM-256M do?

SmolVLM-256M is a capable multimodal AI that can:
- **Describe images**: Generate detailed captions
- **Answer questions about images**: Visual QA
- **Read text in images**: OCR and document understanding
- **Compare images**: Identify differences between photos
- **Process video**: Analyze frames in real-time (via rapid inference)

It is NOT designed for:
- Creative image generation (use Stable Diffusion for that)
- Complex multi-step reasoning
- Tasks requiring very long context (>2K tokens)

### How accurate is SmolVLM-256M?

SmolVLM-256M achieves strong performance for its size:
- **VQAv2**: ~70% accuracy
- **TextVQA**: ~55% accuracy
- **DocVQA**: ~45% accuracy

It's comparable to GPT-4V on simple tasks, but less capable on complex reasoning. It's ideal for high-throughput, latency-sensitive applications where "good enough" answers at extreme speed matter more than maximum accuracy.

### What process node is SiLens manufactured on?

SiLens uses **SkyWater SKY130**, a 130nm CMOS process. Yes, this is old by modern standards (iPhones use 3nm), but:
- It's the only **fully open-source PDK** available
- Mask costs are ~$100K vs. $10M+ for modern nodes
- 130nm is sufficient for our architecture (we're not compute-bound)
- This enables us to make the hardware truly open

Future versions will use more advanced nodes as open PDKs mature.

---

## 2. Performance & Comparisons

### How fast is SiLens?

| Metric | SiLens |
|--------|--------|
| Single-image latency | <5ms |
| Throughput (single stream) | 200+ images/sec |
| Throughput (pipelined) | 1000+ images/sec |

### How does SiLens compare to a GPU?

| Metric | RTX 4060 ($299) | SiLens ($149-249) |
|--------|-----------------|-------------------|
| Latency | 300-1000ms | <5ms |
| Throughput (single) | 1-3 img/sec | 200+ img/sec |
| Throughput (batch) | 5-15 img/sec | 1000+ img/sec |
| Power | 115W | 25W |

SiLens is **60-200× faster** at **1/5th the power** while costing **20-50% less**.

### Why is SiLens so much faster than a GPU?

The RTX 4060 is limited by **memory bandwidth**, not compute. When running SmolVLM-256M:
1. The model (500MB) sits in VRAM
2. For each token, weights are loaded from memory to compute units
3. Memory bandwidth (288 GB/s) becomes the bottleneck
4. GPU compute utilization is <5%

SiLens eliminates this bottleneck entirely — weights are circuits, not data.

### Can I use SiLens for training?

No. SiLens is inference-only. Use GPUs or cloud services for training.

### Can SiLens replace my GPU?

Not entirely. SiLens is optimized for one specific task: running SmolVLM-256M. For:
- Gaming → You still need a GPU
- Training AI models → You still need a GPU
- Running other AI models → You still need a GPU
- Running SmolVLM-256M at insane speed → **SiLens is 100× better**

### What about Google Coral or Intel Movidius?

| Feature | SiLens | Google Coral | Intel Movidius |
|---------|--------|--------------|----------------|
| Multimodal (vision + language) | ✅ | ❌ | ❌ |
| Latency | <5ms | 15-30ms | 30-100ms |
| Open source | ✅ | Partial | ❌ |
| Price | $149-249 | $75-150 | $80-150 |

Coral and Movidius are vision-only. SiLens is the first edge accelerator for vision-language models.

---

## 3. Compatibility & Software

### What operating systems are supported?

| OS | Support Level |
|----|---------------|
| Linux (Ubuntu 20.04+, Debian 11+) | Full support |
| Windows 10/11 | Planned (stretch goal) |
| macOS | Not supported (no PCIe) |

### What programming languages can I use?

- **Python**: Official SDK (`pip install silens`)
- **C/C++**: Native library included
- **Other languages**: Community bindings welcome (Rust, Go, etc.)

### Does SiLens work with popular ML frameworks?

- **Hugging Face Transformers**: Direct integration
- **ONNX Runtime**: Planned integration
- **LangChain**: Compatible via custom provider
- **LlamaIndex**: Compatible via custom provider

### What PCIe slot do I need?

SiLens requires a **PCIe 3.0 x4** slot (or higher). It's compatible with:
- PCIe 3.0 x4, x8, x16 slots
- PCIe 4.0 and 5.0 slots (backward compatible)

It will NOT work in:
- PCIe x1 slots (insufficient bandwidth)
- M.2 slots
- USB ports

### Does SiLens need external power?

No. SiLens draws 25W from the PCIe slot. No additional power cables required.

### Can I use multiple SiLens cards?

Yes! Multiple cards work together for:
- **Higher throughput**: Each card adds 200+ img/sec
- **Redundancy**: Failover if one card has issues
- **Load balancing**: Distribute workloads

Our driver supports up to 8 cards in a single system.

### What about ARM servers?

SiLens requires PCIe, which is available on many ARM platforms. We plan to test and support:
- NVIDIA Jetson (PCIe models)
- Ampere Altra
- AWS Graviton (via Nitro)

Specific ARM support will be confirmed post-silicon.

### Can I use SiLens in a server/data center?

Yes. SiLens is designed for:
- Standard server PCIe slots
- Passive cooling (no fans to fail)
- 24/7 operation
- Remote management via software API

---

## 4. Campaign & Rewards

### What's the funding goal?

**$100,000** — This covers the tape-out costs and initial production run.

### What if the campaign doesn't reach its goal?

All backers are fully refunded. No partial funding.

### What reward tiers are available?

| Tier | Price | What You Get |
|------|-------|--------------|
| Seed | $49 | Name in credits, updates, Discord |
| Maker | $149 | 1× SiLens Core (early bird) |
| Developer | $199 | 1× SiLens Pro + extras |
| Studio | $499 | 3× SiLens Pro + onboarding |
| Enterprise | $1,499 | 10× SiLens Pro + support |
| Pioneer | $4,999 | 25× SiLens Pro + partner perks |
| Visionary | $14,999 | 100× SiLens Pro + strategic partnership |

### What's the difference between Core and Pro?

| Feature | Core ($149) | Pro ($199) |
|---------|-------------|------------|
| SiLens card | ✅ | ✅ |
| Low-profile bracket | ✅ | ✅ |
| Full-height bracket | ❌ | ✅ |
| Premium heatsink | Standard | Upgraded |
| Documentation | Quick-start card | Full printed booklet |
| Support | Standard | Priority queue |

### Is there an early bird discount?

Yes. The first **500 Core** and **300 Pro** backers get early bird pricing:
- Core: $149 (retail $169)
- Pro: $199 (retail $249)

### Can I change my reward tier after backing?

Yes, during the campaign. After the campaign ends, changes can be made through our pledge manager (BackerKit or similar).

### Are there add-on options?

Yes, through our pledge manager after the campaign:
- Extra brackets: $9 each
- Premium heatsink upgrade: $19
- Additional cards: $159-219
- T-shirts, stickers, etc.

### Can I get a refund?

- **During campaign**: Yes, full refund through Kickstarter
- **After campaign, before shipping**: Yes, minus payment processing fees
- **After shipping**: No refunds; warranty process applies

---

## 5. Shipping & Fulfillment

### When will SiLens ship?

**Estimated delivery: March 2028** (approximately 18 months after campaign end)

### Why does it take so long?

Hardware manufacturing is complex:
1. **Design finalization**: 3 months
2. **Tape-out preparation**: 3 months
3. **Fabrication**: 3 months (foundry lead time)
4. **Validation**: 3 months
5. **Production**: 2 months
6. **Shipping**: 1 month

We've built buffer into this timeline for unexpected delays.

### Where do you ship?

Worldwide, with the following shipping costs:

| Region | Additional Cost |
|--------|-----------------|
| USA | Included |
| Canada | +$5 |
| EU | +$15 |
| UK | +$15 |
| Australia/NZ | +$20 |
| Japan/Korea | +$15 |
| Rest of World | +$25 |

### Who handles customs and duties?

Backers are responsible for any import duties, taxes, or customs fees in their country. We'll provide accurate customs declarations to minimize issues.

### Can I change my shipping address?

Yes, through our pledge manager. Please update your address at least 2 months before shipping begins.

### What if my package is lost or damaged?

We'll work with you to resolve any shipping issues:
- **Lost packages**: We'll file a claim and ship a replacement
- **Damaged on arrival**: Photo documentation required, then replacement shipped
- **Refused/returned**: Reship at backer's expense

---

## 6. Company & Team

### Who is behind SiLens?

SiLens is being developed by SiLens Technologies, Inc., a startup focused on democratizing AI hardware through open-source designs.

**Key team members:**
- [Founder Name] — CEO & Chief Architect ([X] years in ASIC design)
- [Name] — VP Engineering ([X] years in physical design)
- [Name] — ML Lead ([X] years in model optimization)

### Where is the company based?

[Location], with remote team members across [regions].

### Is this your first hardware project?

[Adjust based on actual team experience]

The team has collective experience including:
- [X] chips taped out
- [X] years in semiconductor industry
- Previous roles at [companies]

### How can I contact you?

- **General inquiries**: hello@silens.ai
- **Press**: press@silens.ai
- **Technical questions**: Discord community
- **Backer support**: support@silens.ai

### Will you have investors?

We're pursuing a **community-first funding model**. This Kickstarter campaign is our primary funding source. We may seek additional investment for scaling, but backer support is our foundation.

---

## 7. Technical Deep Dive

### What's the full ASIC specification?

| Parameter | Value |
|-----------|-------|
| Model | SmolVLM-256M |
| Total parameters | 246 million |
| Vision encoder | SigLIP-B/16 (93M) |
| Language model | SmolLM2-135M (135M) |
| Multimodal projector | 18M |
| Process | SkyWater SKY130 (130nm) |
| Die size | ~800mm² |
| Metal layers | 5 |
| Core voltage | 1.8V |
| I/O voltage | 3.3V |
| Clock frequency | 100-200 MHz |
| Package | BGA-625 |

### How does the 1-bit quantization work?

SmolVLM-256M uses ternary weights: {-1, 0, +1}. In our silicon:
- +1 → Metal trace to VDD
- -1 → Metal trace to GND
- 0 → No connection (implicit zero)

Activations remain at higher precision (8-bit) to maintain model quality. This is based on research from BitNet and similar 1-bit quantization techniques.

### What's the memory architecture?

SiLens has minimal on-chip memory:
- **Input buffer**: 2MB for image data
- **Intermediate buffers**: 4MB for activations between layers
- **Output buffer**: 512KB for generated tokens

There is NO weight memory — weights are hardwired.

### What's the power breakdown?

| Component | Power |
|-----------|-------|
| Vision encoder | ~8W |
| Language model | ~12W |
| I/O and clocking | ~3W |
| Power regulation | ~2W |
| **Total** | **~25W** |

### What about yield at 800mm²?

800mm² is a large die. Expected yield: 30-50%.

We've mitigated this by:
- Pricing conservatively (assuming 30% yield)
- Implementing redundancy where possible
- Working with SkyWater on process optimization

### Is there ECC or error correction?

Limited. We implement:
- CRC on PCIe data transfers
- Parity checking on critical paths
- Redundancy in lookup tables

The hardwired weights themselves cannot have errors — they're fixed at manufacturing.

### What about thermal management?

The card includes a passive aluminum heatsink rated for continuous operation at 25W in:
- Standard desktop airflow
- Server environments with 0.5+ m/s airflow
- Ambient temperatures up to 35°C

The Pro tier includes an upgraded heatsink for better sustained performance.

---

## 8. Future Plans

### Will there be a Gen 2?

Yes! Our roadmap includes:

| Generation | Timeline | Process | Model | Improvement |
|------------|----------|---------|-------|-------------|
| Gen 1 | 2028 | SKY130 (130nm) | SmolVLM-256M | This campaign |
| Gen 1.5 | 2029 | 65nm | SmolVLM-256M | 2× speed, 50% power |
| Gen 2 | 2030 | 45nm | SmolVLM-500M | 2× model size |

### Will backers get discounts on future products?

Pioneer and Visionary tier backers get **first access** to Gen 2 at early bird pricing.

We're considering loyalty discounts for all backers — details TBD.

### Will you make USB or M.2 versions?

We're exploring:
- **M.2 version**: Lower power, smaller form factor
- **USB version**: External enclosure option

These depend on campaign success and community demand.

### What about different models?

We're researching:
- **Mask-programmable variants**: Different model per production batch
- **Smaller models**: For lower-cost, lower-power applications
- **Specialized models**: OCR-focused, object detection, etc.

Let us know what models you'd like to see!

### Will the open-source design be maintained?

Yes. We're committed to:
- Releasing full RTL within 6 months of shipping
- Maintaining Linux drivers upstream
- Active GitHub repository with issue tracking
- Community contributions welcome

### How can I contribute to the project?

- **Before silicon**: Test simulations, review designs, improve documentation
- **After shipping**: Driver development, SDK improvements, model optimization
- **Always**: Bug reports, use case development, community support

Join our Discord to get involved!

---

## Still Have Questions?

- **Discord**: [discord.gg/silens]
- **Email**: hello@silens.ai
- **Kickstarter comments**: We monitor and respond daily

---

*FAQ Version: 1.0*
*Last Updated: August 2026*
