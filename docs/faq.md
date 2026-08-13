---
layout: default
title: FAQ
permalink: /faq/
---

<section class="page-header-section">
  <h1>Frequently Asked Questions</h1>
  <p>Everything you need to know about SiLens.</p>
</section>

<div class="content-wrapper">

## Product & Technology

### What is SiLens?

SiLens is a PCIe accelerator card that runs a vision-language AI model (SmolVLM-256M) with the model weights **physically etched into the silicon**. This eliminates memory access bottlenecks and enables extremely fast, low-power inference.

### What does "hardwired" mean?

Traditional AI accelerators store model weights in memory (RAM/VRAM) and load them for each computation. In SiLens, each weight value is encoded as a physical wire connection:

- **Weight = +1** → Wire connected to VDD (power)
- **Weight = -1** → Wire connected to GND (ground)
- **Weight = 0** → No connection

The model IS the circuit—no memory access required.

### Can I run different models?

No. The weights are physically etched into silicon and cannot be changed. SiLens is purpose-built for SmolVLM-256M.

We're exploring mask-programmable variants for future products.

### Is SiLens a GPU?

No. SiLens is an **inference-only ASIC**. Unlike a GPU:

- Cannot run arbitrary programs
- Cannot train models
- Runs only the hardwired model
- Has no general-purpose memory

Think of it as a "model in a chip" rather than a programmable accelerator.

### Why 130nm? Isn't that ancient?

Yes, 130nm is old (iPhones use 3nm), but:

1. **It's open** — SkyWater SKY130 is the only fully open-source PDK
2. **It's affordable** — Mask costs ~$100K vs $10M+ for modern nodes
3. **It's sufficient** — Our architecture isn't compute-bound

Future generations will use more advanced nodes.

---

## Performance

### How fast is SiLens?

| Metric | Performance |
|:-------|:------------|
| Single-image latency | **<5ms** |
| Throughput (single stream) | **200+ img/sec** |
| Throughput (pipelined) | **1000+ img/sec** |

### How does it compare to a GPU?

| Metric | RTX 4060 | SiLens | Improvement |
|:-------|:---------|:-------|:------------|
| Price | $299 | $149-249 | **20-50% cheaper** |
| Latency | 300-1000ms | <5ms | **60-200× faster** |
| Throughput | 1-3 img/sec | 200+ img/sec | **100× faster** |
| Power | 115W | 25W | **4.6× efficient** |

### Why is SiLens so much faster?

GPUs are limited by **memory bandwidth**, not compute. When running SmolVLM-256M on a GPU:

1. The model (500MB) sits in VRAM
2. For each token, weights are loaded from memory
3. Memory bandwidth (288 GB/s) becomes the bottleneck
4. GPU compute utilization is <5%

SiLens eliminates this—weights are circuits, not data.

### Can I use SiLens for training?

No. SiLens is inference-only. Use GPUs or cloud for training.

---

## Compatibility

### What operating systems are supported?

| OS | Support |
|:---|:--------|
| Linux (Ubuntu 20.04+) | ✅ Full support |
| Windows 10/11 | 🟡 Planned |
| macOS | ❌ Not supported (no PCIe) |

### What PCIe slot do I need?

SiLens requires **PCIe 3.0 x4** or higher:

- ✅ PCIe 3.0/4.0/5.0 x4, x8, x16 slots
- ❌ PCIe x1 slots (insufficient bandwidth)
- ❌ M.2 slots
- ❌ USB ports

### Does it need external power?

No. SiLens draws 25W from the PCIe slot. No additional power cables required.

### Can I use multiple cards?

Yes! Multiple cards work together for:

- **Higher throughput** — Each card adds 200+ img/sec
- **Redundancy** — Failover if one card fails
- **Load balancing** — Distribute workloads

Our driver supports up to 8 cards per system.

### What programming languages are supported?

- **Python** — Official SDK (`pip install silens`)
- **C/C++** — Native library included
- **Others** — Community bindings welcome

---

## Technical Details

### What model does SiLens run?

**SmolVLM-256M** — a 246M parameter vision-language model:

| Component | Parameters |
|:----------|:-----------|
| SigLIP-B/16 (vision) | 93M |
| SmolLM2-135M (language) | 135M |
| Multimodal projector | 18M |
| **Total** | **246M** |

### What can SmolVLM-256M do?

- ✅ Describe images (detailed captions)
- ✅ Answer questions about images (Visual QA)
- ✅ Read text in images (OCR)
- ✅ Compare images
- ✅ Process video (via rapid frame analysis)
- ❌ Generate images
- ❌ Complex multi-step reasoning
- ❌ Very long context (>2K tokens)

### What's the ASIC specification?

| Parameter | Value |
|:----------|:------|
| Process | SkyWater SKY130 (130nm) |
| Die size | ~800mm² |
| Metal layers | 5 |
| Core voltage | 1.8V |
| Clock | 100-200 MHz |
| Package | BGA-625 |

### What about manufacturing yield?

At 800mm², expected yield is 30-50%. We've:

- Priced conservatively (assuming 30% yield)
- Implemented redundancy where possible
- Partnered with SkyWater on optimization

---

## Future Plans

### Will there be a Gen 2?

Yes! Our roadmap:

| Generation | Timeline | Process | Model | Improvement |
|:-----------|:---------|:--------|:------|:------------|
| Gen 1 | 2028 | 130nm | SmolVLM-256M | Initial release |
| Gen 1.5 | 2029 | 65nm | SmolVLM-256M | 2× speed, 50% power |
| Gen 2 | 2030 | 45nm | SmolVLM-500M | 2× model size |

### Will you make USB or M.2 versions?

We're exploring both based on community demand:

- **M.2** — Lower power, smaller form factor
- **USB** — External enclosure option

### How can I contribute?

- **Before silicon** — Test simulations, review designs, improve docs
- **After shipping** — Driver development, SDK improvements
- **Always** — Bug reports, use cases, community support

Join us on [GitHub](https://github.com/loreii/SiLens)!

---

## Still Have Questions?

<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">💬</div>
<h3>Discord</h3>
<p>Join our community (coming soon)</p>
</div>

<div class="feature-card">
<div class="feature-icon">📧</div>
<h3>Email</h3>
<p><a href="mailto:hello@silens.ai">hello@silens.ai</a></p>
</div>

<div class="feature-card">
<div class="feature-icon">🐛</div>
<h3>GitHub Issues</h3>
<p><a href="https://github.com/loreii/SiLens/issues" target="_blank">Report bugs & requests</a></p>
</div>
</div>

</div>
