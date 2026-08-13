---
layout: default
title: Getting Started
permalink: /getting-started/
---

<section class="page-header-section">
  <h1>Getting Started</h1>
  <p>Set up your development environment and start exploring SiLens.</p>
</section>

<div class="content-wrapper">

<div class="alert alert-info">
  <span class="alert-icon">ℹ️</span>
  <div>
    <strong>Note:</strong> SiLens hardware is not yet available. These instructions help you work with the simulation and model conversion tools.
  </div>
</div>

## Quick Start: Interactive Demo

The fastest way to explore SiLens capabilities:

```bash
git clone https://github.com/loreii/SiLens.git
cd SiLens
pip install numpy
python demo.py
```

The demo includes:

| Demo | Description |
|:-----|:------------|
| Ternary Quantization | Convert FP32 weights to 2-bit ternary (16× compression) |
| Hardware Simulation | Interact with simulated SiLens accelerator |
| Performance Profiling | Detailed timing and throughput analysis |
| Multi-Device Inference | Distributed batch processing |
| Sparse Attention | Attention pattern optimization |
| End-to-End Pipeline | Complete inference demonstration |

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** with pip
- **Git** for cloning the repository
- **8GB+ RAM** recommended for model conversion
- **NVIDIA GPU** (optional) for faster analysis

---

## Full Installation

### 1. Clone Repository

```bash
git clone https://github.com/loreii/SiLens.git
cd SiLens
git submodule update --init --recursive
```

### 2. Create Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Model Weights

```bash
python tools/download_model.py
```

Or manually:

```bash
pip install transformers torch pillow

python -c "
from transformers import AutoProcessor, AutoModelForVision2Seq
processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct')
model = AutoModelForVision2Seq.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct')
processor.save_pretrained('model/smolvlm-256m')
model.save_pretrained('model/smolvlm-256m')
"
```

---

## Model Quantization Workflow

Convert the FP32 model to ternary weights for hardware implementation.

### Step 1: Analyze Model

```bash
python model/conversion/analyze_model.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct
```

### Step 2: Sensitivity Analysis

Identify layers most sensitive to quantization:

```bash
python model/conversion/sensitivity_analysis.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct
```

### Step 3: Quantize to Ternary

```bash
python model/conversion/quantize_ternary.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --alpha 0.7 \
    --mode per_tensor \
    --export \
    --output ./model/weights/quantized
```

### Step 4: Validate Quality

```bash
python model/conversion/validate_quantization.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --quantized ./model/weights/quantized \
    --detailed
```

### Step 5: Run Benchmarks

```bash
python model/validation/benchmark_suite.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --all
```

---

## Optimized Quantization

For best accuracy, use gradient-based alpha optimization:

```bash
python model/conversion/quantize_ternary.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --optimize-alpha \
    --export \
    --output ./model/weights/optimized
```

---

## Expected Results

With default settings (α=0.7, per_tensor):

| Metric | Original | Quantized | Change |
|:-------|:---------|:----------|:-------|
| Memory | 1024 MB | 64 MB | **16× smaller** |
| VQA Accuracy | ~71% | ~67% | ~4% drop |
| Perplexity | ~15 | ~17 | ~13% increase |
| Cosine Similarity | — | 0.92 | — |

---

## Repository Structure

```
SiLens/
├── model/
│   ├── conversion/      # Quantization tools
│   ├── validation/      # Accuracy benchmarks
│   └── analysis/        # Weight visualization
├── rtl/                 # Verilog source (coming soon)
├── fpga/                # FPGA prototypes
├── drivers/             # Linux kernel driver
├── sdk/                 # Python SDK
├── firmware/            # Card firmware
└── docs/                # Documentation
```

---

## Next Steps

<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">📐</div>
<h3>Architecture</h3>
<p>Understand the hardware design and data flow.</p>
<a href="{{ site.baseurl }}/architecture/">Learn more →</a>
</div>

<div class="feature-card">
<div class="feature-icon">📖</div>
<h3>Documentation</h3>
<p>Detailed API reference and tool documentation.</p>
<a href="{{ site.baseurl }}/docs/">View docs →</a>
</div>

<div class="feature-card">
<div class="feature-icon">🤝</div>
<h3>Contributing</h3>
<p>Join the community and help build SiLens.</p>
<a href="https://github.com/loreii/SiLens/blob/main/CONTRIBUTING.md" target="_blank">Contribute →</a>
</div>
</div>

</div>
