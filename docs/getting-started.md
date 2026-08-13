---
layout: default
title: Getting Started
permalink: /getting-started/
---

# Getting Started with SiLens

## Prerequisites

Before you begin, ensure you have the following:

- **Python 3.8+** with pip
- **Git** for cloning the repository
- **8GB+ RAM** recommended for model conversion tools
- **NVIDIA GPU** (optional) for faster model analysis

---

## Quick Start: Interactive Demo

The fastest way to explore SiLens capabilities:

```bash
git clone https://github.com/loreii/SiLens.git
cd SiLens
pip install numpy
python demo.py
```

The demo showcases:

| Demo | Description |
|------|-------------|
| **1. Ternary Quantization** | Convert FP32 weights to 2-bit ternary (16x compression) |
| **2. Hardware Simulation** | Interact with simulated SiLens accelerator |
| **3. Performance Profiling** | Detailed timing and throughput analysis |
| **4. Multi-Device Inference** | Distributed batch processing |
| **5. Sparse Attention** | Attention pattern optimization for hardware |
| **6. End-to-End Pipeline** | Complete inference demonstration |

---

## Full Installation

### 1. Clone Repository

```bash
git clone https://github.com/loreii/SiLens.git
cd SiLens
git submodule update --init --recursive
```

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
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

The complete model conversion pipeline from FP32 to ternary weights:

### Step 1: Analyze Model Architecture

```bash
python model/conversion/analyze_model.py --model HuggingFaceTB/SmolVLM-256M-Instruct
```

### Step 2: Run Sensitivity Analysis

Identify which layers are most sensitive to quantization:

```bash
python model/conversion/sensitivity_analysis.py --model HuggingFaceTB/SmolVLM-256M-Instruct
```

### Step 3: Quantize to Ternary Weights

```bash
python model/conversion/quantize_ternary.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --alpha 0.7 \
    --mode per_tensor \
    --export \
    --output ./model/weights/quantized
```

### Step 4: Validate Quantization Quality

```bash
python model/conversion/validate_quantization.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --quantized ./model/weights/quantized \
    --detailed
```

### Step 5: Run Accuracy Benchmarks

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

With default settings (α=0.7, per_tensor quantization):

| Metric | Original | Quantized | Change |
|--------|----------|-----------|--------|
| Memory | 1024 MB | 64 MB | **16× reduction** |
| VQA Accuracy | ~71% | ~67% | ~4% drop |
| Perplexity | ~15 | ~17 | ~13% increase |
| Cosine Similarity | - | 0.92 | - |

---

## Running Tests

### RTL Simulation (when available)

```bash
make sim
```

### Python Tests

```bash
pytest tests/
```

---

## Dependencies

### SkyWater SKY130 PDK

The open-source process design kit for 130nm CMOS fabrication:

```bash
# Clone the PDK (large download, ~7GB)
git clone https://github.com/google/skywater-pdk.git pdk/skywater-pdk

# Or add as submodule
git submodule add https://github.com/google/skywater-pdk.git pdk/skywater-pdk
```

**Documentation:** [skywater-pdk.readthedocs.io](https://skywater-pdk.readthedocs.io/)

### OpenLane (Synthesis)

RTL-to-GDSII flow using open-source tools:

```bash
# Install via Docker (recommended)
docker pull efabless/openlane:latest
```

**Documentation:** [openlane.readthedocs.io](https://openlane.readthedocs.io/)

---

## Repository Structure

```
SiLens/
├── README.md                 # Main readme
├── LICENSE                   # Apache 2.0
├── docs/                     # Documentation (this site)
├── rtl/                      # Verilog/SystemVerilog source
│   ├── vision_encoder/       # SigLIP-B/16 implementation
│   ├── language_model/       # SmolLM2-135M implementation
│   ├── projector/            # Multimodal projector
│   ├── top/                  # Top-level integration
│   └── tb/                   # Testbenches
├── model/                    # Model files and conversion tools
│   ├── weights/              # Quantized weights (gitignored)
│   ├── conversion/           # PyTorch → Verilog tools
│   ├── validation/           # Model accuracy validation
│   └── analysis/             # Weight analysis tools
├── pdk/                      # SkyWater PDK setup
├── synthesis/                # OpenLane synthesis scripts
├── pcb/                      # PCB design files
├── firmware/                 # Card firmware
├── drivers/                  # Linux kernel driver
├── sdk/                      # Python SDK
├── fpga/                     # FPGA prototype files
└── tools/                    # Utility scripts
```

---

## Next Steps

- [📐 Architecture Overview]({{ site.baseurl }}/architecture/) - Understand the hardware design
- [📖 Documentation]({{ site.baseurl }}/docs/) - Detailed technical documentation
- [❓ FAQ]({{ site.baseurl }}/faq/) - Frequently asked questions
- [🤝 Contributing](https://github.com/loreii/SiLens/blob/main/CONTRIBUTING.md) - How to contribute

---

[← Back to Home]({{ site.baseurl }}/)
