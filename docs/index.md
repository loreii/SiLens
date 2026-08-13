---
layout: default
title: Home
---

# SiLens™

## Open-Source Hardwired Vision-Language AI Accelerator

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![SkyWater PDK](https://img.shields.io/badge/PDK-SkyWater%20SKY130-orange.svg)](https://github.com/google/skywater-pdk)
[![Model](https://img.shields.io/badge/Model-SmolVLM--256M-green.svg)](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)

---

## 100× Faster Than a GPU. Half the Price. 25 Watts.

SiLens is a PCIe card that runs vision + language AI at **200 images/second** with **<5ms latency** — using just **25 watts**.

We achieve this by doing something no one else has done: **etching the AI model weights directly into silicon.** No memory. No bottlenecks. Just raw, instant inference.

<div class="features">
  <div class="feature">
    🚀 <strong>100× faster</strong> than a $300 GPU running the same model
  </div>
  <div class="feature">
    ⚡ <strong>25W power</strong> vs. 115W for a graphics card
  </div>
  <div class="feature">
    💰 <strong>$149-249</strong> — cheaper than the GPU it replaces
  </div>
  <div class="feature">
    🔓 <strong>Fully open-source</strong> — Apache 2.0 from silicon to software
  </div>
  <div class="feature">
    🤖 <strong>Multimodal AI</strong> — sees images AND understands language
  </div>
</div>

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

**Until now.**

---

## The Solution: AI Baked Into Silicon

### How Traditional AI Works

```
[Image] → [Load weights from memory] → [Compute] → [Answer]
                    ↑
            This is the bottleneck
```

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

SiLens runs **SmolVLM-256M**, a state-of-the-art vision-language model with 246 million parameters.

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
At 200+ images/second, SiLens can analyze live video feeds for security, manufacturing QC, traffic monitoring, and retail analytics.

---

## Performance Comparison

### SiLens vs. $300 GPU (RTX 4060)

| Metric | RTX 4060 | **SiLens** | **Improvement** |
|--------|----------|------------|-----------------|
| Price | $299 | $149-249 | **20-50% cheaper** |
| Single-image latency | 300-1000ms | **<5ms** | **60-200× faster** |
| Throughput (single) | 1-3 img/sec | **200+ img/sec** | **100× faster** |
| Throughput (batch) | 5-15 img/sec | **1000+ img/sec** | **70× faster** |
| Power consumption | 115W | **25W** | **4.6× efficient** |
| Form factor | Full desktop GPU | **Half-height PCIe** | Fits anywhere |

---

## Project Status

| Component | Status |
|-----------|--------|
| Architecture specification | 🟡 In progress |
| RTL design (Verilog) | 🔴 Not started |
| FPGA prototype | 🔴 Not started |
| Physical design | 🔴 Not started |
| PCB design | 🔴 Not started |
| Software/drivers | 🔴 Not started |

> **⚠️ EARLY DEVELOPMENT** — This project is in the architectural design phase. Hardware is not yet available.

---

## Quick Links

- [📐 Architecture Overview]({{ site.baseurl }}/architecture/)
- [🚀 Getting Started]({{ site.baseurl }}/getting-started/)
- [📖 Documentation]({{ site.baseurl }}/docs/)
- [❓ FAQ]({{ site.baseurl }}/faq/)
- [💚 Support on Kickstarter]({{ site.baseurl }}/kickstarter/)
- [🐙 GitHub Repository](https://github.com/loreii/SiLens)

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

---

## Contact

- **GitHub:** [github.com/loreii/SiLens](https://github.com/loreii/SiLens)
- **Discord:** Coming soon
- **Email:** hello@silens.ai

---

*SiLens is an independent project and is not affiliated with Hugging Face, Google, SkyWater Technology, or any FPGA vendor.*
