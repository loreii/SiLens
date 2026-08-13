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

A PCIe accelerator card that runs SmolVLM-256M with weights **physically etched into silicon**. This eliminates memory bottlenecks for extremely fast, low-power inference.

### What does "hardwired" mean?

Traditional accelerators store weights in memory and load them for each computation. In SiLens, each weight is a physical wire:

- **Weight = +1** → Wire to VDD (power)
- **Weight = -1** → Wire to GND (ground)
- **Weight = 0** → No connection

The model IS the circuit—no memory access needed.

### Can I run different models?

No. Weights are physically etched and cannot be changed. SiLens is purpose-built for SmolVLM-256M.

### Is SiLens a GPU?

No. It's an **inference-only ASIC**:
- Cannot run arbitrary programs
- Cannot train models
- Runs only the hardwired model
- Has no general-purpose memory

### Why 130nm?

1. **It's open** — SkyWater SKY130 is the only fully open-source PDK
2. **It's affordable** — ~$100K masks vs $10M+ for modern nodes
3. **It's sufficient** — Our architecture isn't compute-bound

---

## Performance

### How fast is SiLens?

- **Latency:** <5ms single image
- **Throughput:** 200+ images/sec
- **Pipelined:** 1000+ images/sec

### Compared to RTX 4060?

- **Price:** $149-249 vs $299 (20-50% cheaper)
- **Latency:** <5ms vs 300-1000ms (60-200× faster)
- **Throughput:** 200+ vs 1-3 img/sec (100× faster)
- **Power:** 25W vs 115W (4.6× more efficient)

### Why so much faster?

GPUs are limited by memory bandwidth, not compute. When running SmolVLM-256M:
- Model (500MB) sits in VRAM
- Weights loaded from memory each token
- Memory bandwidth is the bottleneck
- GPU compute utilization <5%

SiLens eliminates this—weights are circuits, not data.

### Can I train on SiLens?

No. Inference only. Use GPUs/cloud for training.

---

## Compatibility

### Operating Systems?

- **Linux (Ubuntu 20.04+):** ✅ Full support
- **Windows 10/11:** 🟡 Planned
- **macOS:** ❌ Not supported (no PCIe)

### What PCIe slot?

Requires **PCIe 3.0 x4** or higher:
- ✅ x4, x8, x16 slots (3.0/4.0/5.0)
- ❌ x1 slots
- ❌ M.2 slots
- ❌ USB

### External power needed?

No. 25W from PCIe slot. No cables needed.

### Multiple cards?

Yes! Up to 8 cards per system for:
- Higher throughput
- Redundancy
- Load balancing

### Programming languages?

- **Python:** Official SDK (`pip install silens`)
- **C/C++:** Native library
- **Others:** Community bindings welcome

---

## Technical Details

### What model?

**SmolVLM-256M** (246M parameters):
- Vision: SigLIP-B/16 (93M)
- Language: SmolLM2-135M (135M)
- Projector: 18M

### What can it do?

✅ Describe images
✅ Answer questions about images
✅ Read text in images (OCR)
✅ Compare images
✅ Process video frames

❌ Generate images
❌ Complex reasoning
❌ Long context (>2K tokens)

### ASIC specs?

- **Process:** SkyWater SKY130 (130nm)
- **Die size:** ~800mm²
- **Voltage:** 1.8V core, 3.3V I/O
- **Clock:** 100-200 MHz
- **Package:** BGA-625

### Manufacturing yield?

At 800mm², expected 30-50% yield. We've:
- Priced conservatively (30% assumption)
- Added redundancy where possible
- Partnered with SkyWater on optimization

---

## Future Plans

### Gen 2?

Yes! Roadmap:
- **Gen 1 (2028):** 130nm, SmolVLM-256M
- **Gen 1.5 (2029):** 65nm, 2× speed, 50% power
- **Gen 2 (2030):** 45nm, SmolVLM-500M

### USB or M.2 versions?

Exploring based on demand:
- **M.2:** Lower power, smaller
- **USB:** External enclosure

### How to contribute?

- Test simulations
- Review designs
- Improve documentation
- Driver development
- Bug reports

Join us on [GitHub](https://github.com/loreii/SiLens)!

---

## Contact

<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">💬</div>
<h3>Discord</h3>
<p>Coming soon</p>
</div>

<div class="feature-card">
<div class="feature-icon">📧</div>
<h3>Email</h3>
<p><a href="mailto:hello@silens.ai">hello@silens.ai</a></p>
</div>

<div class="feature-card">
<div class="feature-icon">🐛</div>
<h3>GitHub</h3>
<p><a href="https://github.com/loreii/SiLens/issues" target="_blank">Issues & Requests</a></p>
</div>
</div>

</div>
