---
layout: default
title: Documentation
permalink: /docs/
---

<section class="page-header-section">
  <h1>Documentation</h1>
  <p>Technical guides for working with SiLens hardware and software.</p>
</section>

<div class="content-wrapper">

## Core Documentation

<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">📐</div>
<h3>Architecture Overview</h3>
<p>System architecture, data flow, and hardware specifications.</p>
<a href="{{ site.baseurl }}/architecture/">Read more →</a>
</div>

<div class="feature-card">
<div class="feature-icon">🚀</div>
<h3>Getting Started</h3>
<p>Installation, setup, and your first steps with SiLens.</p>
<a href="{{ site.baseurl }}/getting-started/">Read more →</a>
</div>

<div class="feature-card">
<div class="feature-icon">📖</div>
<h3>Quantization Guide</h3>
<p>Complete guide to ternary quantization for SiLens.</p>
<a href="https://github.com/loreii/SiLens/blob/main/model/QUANTIZATION_GUIDE.md" target="_blank">Read more →</a>
</div>
</div>

---

## Model Conversion Tools

The `model/conversion/` directory contains tools for quantizing SmolVLM-256M.

### Quantization Pipeline

| Tool | Purpose |
|:-----|:--------|
| `analyze_model.py` | Architecture analysis, weight statistics |
| `extract_weights.py` | Extract and organize weights |
| `quantize_ternary.py` | Ternary quantization with multiple modes |
| `calibration.py` | Calibration-aware quantization |
| `mixed_precision.py` | Keep critical layers at higher precision |
| `sensitivity_analysis.py` | Layer-by-layer sensitivity scoring |

### Validation Tools

| Tool | Purpose |
|:-----|:--------|
| `validate_quantization.py` | Layer-by-layer quality validation |
| `benchmark_suite.py` | VQA, TextVQA, captioning benchmarks |
| `perplexity_test.py` | Language model perplexity measurement |
| `visual_qa_test.py` | Visual QA accuracy testing |
| `compare_outputs.py` | Side-by-side output comparison |

### Analysis Tools

| Tool | Purpose |
|:-----|:--------|
| `weight_visualizer.py` | Distribution plots with matplotlib |
| `sparsity_analyzer.py` | Sparsity patterns and structured sparsity |
| `outlier_detector.py` | Identify and handle outlier weights |

---

## Hardware Documentation

### FPGA Prototyping

The `fpga/` directory contains prototype files:

**Xilinx:**
- `silens_fpga_wrapper.v` — Top-level FPGA wrapper
- `silens_artix7.xdc` — Artix-7 constraints
- `silens_kintex7.xdc` — Kintex-7 constraints
- `synth_vivado.tcl` — Vivado synthesis script

**Intel:**
- `silens_fpga_wrapper_intel.v` — Intel FPGA wrapper
- `silens_arria10.sdc` — Arria 10 constraints
- `silens_cyclone10.sdc` — Cyclone 10 constraints

### PCB Design

PCB design files in `pcb/` (coming soon):
- Schematics
- Layout files
- Bill of Materials (BOM)
- Assembly instructions

---

## Software Documentation

### Linux Kernel Driver

The `drivers/` directory contains:
- `silens_drv.c` — Main driver source
- `silens_ioctl.h` — IOCTL definitions
- `Makefile` — Build instructions

### Python SDK

```bash
pip install silens
```

```python
import silens

# Initialize the device
device = silens.Device()

# Load an image
from PIL import Image
image = Image.open("photo.jpg")

# Run inference
result = device.describe(image)
print(result)

# Visual QA
answer = device.ask(image, "What color is the car?")
print(answer)
```

### Firmware

The `firmware/` directory contains:
- `main.c` — Main firmware application
- `startup.S` — Startup assembly code
- `linker.ld` — Linker script

---

## API Reference

### Device API

```python
class silens.Device:
    def __init__(self, device_id: int = 0)
    def describe(self, image: PIL.Image) -> str
    def ask(self, image: PIL.Image, question: str) -> str
    def batch_describe(self, images: List[PIL.Image]) -> List[str]
    def get_info(self) -> DeviceInfo
    def close(self) -> None
```

### DeviceInfo

```python
class silens.DeviceInfo:
    device_id: int
    firmware_version: str
    temperature: float
    power_draw: float
    utilization: float
```

---

## Technical Specifications

### The ASIC

| Specification | Value |
|:--------------|:------|
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
|:--------------|:------|
| Interface | PCIe 3.0 x4 |
| Form Factor | Half-height, half-length |
| Dimensions | 168mm × 69mm |
| Power | 25W TDP (slot-powered) |
| Cooling | Passive heatsink |
| Weight | ~150g |

### Software Support

| Platform | Status |
|:---------|:-------|
| Linux | Full support (kernel driver + Python API) |
| Windows | Community support planned |
| macOS | Not supported (no PCIe) |
| Docker | Official container images |

---

## Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/loreii/SiLens/blob/main/CONTRIBUTING.md).

### Areas Needing Help

- RTL design for transformer blocks
- FPGA prototyping
- PCB design review
- Driver development
- Documentation

---

## License

Apache License 2.0 — see [LICENSE](https://github.com/loreii/SiLens/blob/main/LICENSE).

### Third-Party Licenses

| Component | License | Source |
|:----------|:--------|:-------|
| SmolVLM-256M | Apache 2.0 | Hugging Face |
| SkyWater SKY130 | Apache 2.0 | Google/SkyWater |
| OpenLane | Apache 2.0 | Efabless |

</div>
