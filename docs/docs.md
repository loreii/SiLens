---
layout: default
title: Documentation
permalink: /docs/
---

<section class="page-header-section">
  <h1>Documentation</h1>
  <p>Technical guides for SiLens hardware and software.</p>
</section>

<div class="content-wrapper">

## Core Documentation

<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">📐</div>
<h3>Architecture</h3>
<p>System design and specifications.</p>
<a href="{{ site.baseurl }}/architecture/">Read more →</a>
</div>

<div class="feature-card">
<div class="feature-icon">🚀</div>
<h3>Getting Started</h3>
<p>Installation and setup.</p>
<a href="{{ site.baseurl }}/getting-started/">Read more →</a>
</div>

<div class="feature-card">
<div class="feature-icon">📖</div>
<h3>Quantization Guide</h3>
<p>Ternary quantization for SiLens.</p>
<a href="https://github.com/loreii/SiLens/blob/main/model/QUANTIZATION_GUIDE.md" target="_blank">Read more →</a>
</div>
</div>

---

## Model Conversion Tools

Located in `model/conversion/`:

### Quantization Pipeline

- **analyze_model.py** — Architecture analysis, weight statistics
- **extract_weights.py** — Extract and organize weights
- **quantize_ternary.py** — Ternary quantization
- **calibration.py** — Calibration-aware quantization
- **mixed_precision.py** — Keep critical layers higher precision
- **sensitivity_analysis.py** — Layer sensitivity scoring

### Validation Tools

- **validate_quantization.py** — Quality validation
- **benchmark_suite.py** — VQA, TextVQA benchmarks
- **perplexity_test.py** — Language model perplexity
- **visual_qa_test.py** — Visual QA accuracy
- **compare_outputs.py** — Side-by-side comparison

### Analysis Tools

- **weight_visualizer.py** — Distribution plots
- **sparsity_analyzer.py** — Sparsity patterns
- **outlier_detector.py** — Outlier detection

---

## Hardware Documentation

### FPGA Prototyping

Located in `fpga/`:

**Xilinx:**
- silens_fpga_wrapper.v
- silens_artix7.xdc
- silens_kintex7.xdc
- synth_vivado.tcl

**Intel:**
- silens_fpga_wrapper_intel.v
- silens_arria10.sdc
- silens_cyclone10.sdc

### PCB Design

Located in `pcb/` (coming soon):
- Schematics
- Layout files
- Bill of Materials
- Assembly instructions

---

## Software Documentation

### Linux Driver

Located in `drivers/`:
- **silens_drv.c** — Main driver
- **silens_ioctl.h** — IOCTL definitions
- **Makefile** — Build instructions

### Python SDK

```bash
pip install silens
```

```python
import silens

device = silens.Device()

from PIL import Image
image = Image.open("photo.jpg")

# Describe image
result = device.describe(image)

# Visual QA
answer = device.ask(image, "What color is the car?")
```

### Firmware

Located in `firmware/`:
- **main.c** — Main application
- **startup.S** — Startup code
- **linker.ld** — Linker script

---

## Specifications

### ASIC

- **Model:** SmolVLM-256M (246M parameters)
- **Vision:** SigLIP-B/16 (93M)
- **Language:** SmolLM2-135M (135M)
- **Process:** SkyWater SKY130 (130nm)
- **Die Size:** ~800mm²
- **Clock:** 100-200 MHz

### Card

- **Interface:** PCIe 3.0 x4
- **Form Factor:** Half-height, half-length
- **Dimensions:** 168mm × 69mm
- **Power:** 25W TDP (slot-powered)
- **Cooling:** Passive heatsink

### Software Support

- **Linux:** Full support
- **Windows:** Planned
- **macOS:** Not supported (no PCIe)
- **Docker:** Official images

---

## Contributing

See [CONTRIBUTING.md](https://github.com/loreii/SiLens/blob/main/CONTRIBUTING.md)

### Areas Needing Help

- RTL design for transformer blocks
- FPGA prototyping
- PCB design review
- Driver development
- Documentation

---

## License

Apache License 2.0

### Third-Party

- **SmolVLM-256M** — Apache 2.0 (Hugging Face)
- **SkyWater SKY130** — Apache 2.0 (Google/SkyWater)
- **OpenLane** — Apache 2.0 (Efabless)

</div>
