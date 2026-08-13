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

## Quick Start

The fastest way to explore SiLens:

```bash
git clone https://github.com/loreii/SiLens.git
cd SiLens
pip install numpy
python demo.py
```

### Demo Features

- **Ternary Quantization** — Convert FP32 weights to 2-bit (16× compression)
- **Hardware Simulation** — Interact with simulated accelerator
- **Performance Profiling** — Timing and throughput analysis
- **Multi-Device Inference** — Distributed batch processing
- **Sparse Attention** — Attention pattern optimization
- **End-to-End Pipeline** — Complete inference demo

---

## Prerequisites

- **Python 3.8+** with pip
- **Git** for cloning
- **8GB+ RAM** recommended
- **NVIDIA GPU** (optional, for faster analysis)

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
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Model

```bash
python tools/download_model.py
```

---

## Model Quantization

Convert FP32 model to ternary weights:

### Step 1: Analyze

```bash
python model/conversion/analyze_model.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct
```

### Step 2: Sensitivity Analysis

```bash
python model/conversion/sensitivity_analysis.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct
```

### Step 3: Quantize

```bash
python model/conversion/quantize_ternary.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --alpha 0.7 --mode per_tensor \
    --export --output ./model/weights/quantized
```

### Step 4: Validate

```bash
python model/conversion/validate_quantization.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --quantized ./model/weights/quantized
```

---

## Expected Results

With default settings (α=0.7):

- **Memory:** 1024 MB → 64 MB (16× smaller)
- **VQA Accuracy:** ~71% → ~67% (~4% drop)
- **Perplexity:** ~15 → ~17 (~13% increase)
- **Cosine Similarity:** 0.92

---

## Repository Structure

- **model/** — Quantization and validation tools
- **rtl/** — Verilog source (coming soon)
- **fpga/** — FPGA prototypes
- **drivers/** — Linux kernel driver
- **sdk/** — Python SDK
- **firmware/** — Card firmware
- **docs/** — Documentation

---

## Next Steps

<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">📐</div>
<h3>Architecture</h3>
<p>Understand the hardware design.</p>
<a href="{{ site.baseurl }}/architecture/">Learn more →</a>
</div>

<div class="feature-card">
<div class="feature-icon">📖</div>
<h3>Documentation</h3>
<p>API reference and tools.</p>
<a href="{{ site.baseurl }}/docs/">View docs →</a>
</div>

<div class="feature-card">
<div class="feature-icon">🤝</div>
<h3>Contributing</h3>
<p>Help build SiLens.</p>
<a href="https://github.com/loreii/SiLens/blob/main/CONTRIBUTING.md" target="_blank">Contribute →</a>
</div>
</div>

</div>
