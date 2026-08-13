---
layout: default
title: FAQ
permalink: /faq/
---

# Frequently Asked Questions

## Table of Contents

1. [Product & Technology](#product--technology)
2. [Performance & Comparisons](#performance--comparisons)
3. [Compatibility & Software](#compatibility--software)
4. [Technical Details](#technical-details)
5. [Future Plans](#future-plans)

---

## Product & Technology

### What is SiLens?

SiLens is a PCIe accelerator card that runs a vision-language AI model (SmolVLM-256M) with the model weights physically etched into the silicon. This eliminates memory access bottlenecks and enables extremely fast, low-power inference.

### What does "hardwired" mean?

Traditional AI accelerators store model weights in memory (RAM/VRAM) and load them for each computation. In SiLens, each weight value is encoded as a physical wire connection in the chip:
- **Weight = +1** → Wire connected to VDD (power)
- **Weight = -1** → Wire connected to GND (ground)

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

### Why 130nm? Isn't that ancient?

Yes, 130nm is old by modern standards (iPhones use 3nm), but:
1. **It's open** — SkyWater SKY130 is the only fully open-source PDK
2. **It's cheap** — Mask costs are ~$100K vs. $10M+ for modern nodes
3. **It's enough** — Our architecture doesn't need bleeding-edge transistors

Future generations will move to 65nm and beyond as open PDKs mature.

---

## Performance & Comparisons

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

## Compatibility & Software

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

---

## Technical Details

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

---

## Future Plans

### Will there be a Gen 2?

Yes! Our roadmap includes:

| Generation | Timeline | Process | Model | Improvement |
|------------|----------|---------|-------|-------------|
| Gen 1 | 2028 | SKY130 (130nm) | SmolVLM-256M | This campaign |
| Gen 1.5 | 2029 | 65nm | SmolVLM-256M | 2× speed, 50% power |
| Gen 2 | 2030 | 45nm | SmolVLM-500M | 2× model size |

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

- **Discord**: Coming soon
- **Email**: hello@silens.ai
- **GitHub Issues**: [github.com/loreii/SiLens/issues](https://github.com/loreii/SiLens/issues)

---

[← Back to Home]({{ site.baseurl }}/)
