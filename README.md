# SiLens™ — Open-Source Hardwired Vision-Language AI Accelerator

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![SkyWater PDK](https://img.shields.io/badge/PDK-SkyWater%20SKY130-orange.svg)](https://github.com/google/skywater-pdk)
[![Model](https://img.shields.io/badge/Model-SmolVLM--256M-green.svg)](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)

> **⚠️ EARLY DEVELOPMENT** — This project is in the architectural design phase. Hardware is not yet available.

## What is SiLens?

SiLens is an open-source AI accelerator that implements a vision-language model (SmolVLM-256M) with weights **hardwired directly into silicon**. Instead of storing neural network weights in memory, each weight becomes a physical wire connection:

- **Weight = +1** → Wire to VDD (power)
- **Weight = -1** → Wire to GND (ground)

This eliminates the memory bottleneck that limits traditional AI accelerators, enabling:
- **<5ms latency** (vs 300-1000ms on GPU)
- **200+ images/second** throughput
- **25W power** (vs 115W for comparable GPU)

## Project Status

| Component | Status |
|-----------|--------|
| Architecture specification | 🟡 In progress |
| RTL design (Verilog) | 🟢 Core modules complete |
| RTL simulation | 🟢 E2E pipeline verified |
| FPGA prototype | 🔴 Not started |
| Physical design | 🔴 Not started |
| PCB design | 🔴 Not started |
| Software/drivers | 🟡 SDK in progress |

## Repository Structure

```
SiLens/
├── README.md                 # This file
├── LICENSE                   # Apache 2.0
├── docs/                     # Documentation
│   ├── architecture/         # System architecture
│   ├── business/            # Business plan documents
│   └── kickstarter/         # Crowdfunding materials
├── rtl/                     # Verilog/SystemVerilog source
│   ├── vision_encoder/      # SigLIP-B/16 implementation
│   ├── language_model/      # SmolLM2-135M implementation
│   ├── projector/           # Multimodal projector
│   ├── top/                 # Top-level integration
│   └── tb/                  # Testbenches
├── model/                   # Model files and conversion tools
│   ├── weights/             # Quantized weights (gitignored)
│   ├── conversion/          # PyTorch → Verilog tools
│   │   ├── quantize_ternary.py    # Ternary quantization with gradient optimization
│   │   ├── calibration.py         # Calibration-aware quantization
│   │   ├── mixed_precision.py     # Mixed-precision quantization
│   │   └── sensitivity_analysis.py # Layer sensitivity analysis
│   ├── validation/          # Model accuracy validation
│   │   ├── benchmark_suite.py     # VQA/TextVQA benchmarks
│   │   ├── perplexity_test.py     # Language model perplexity
│   │   ├── visual_qa_test.py      # Visual QA accuracy
│   │   └── compare_outputs.py     # Original vs quantized comparison
│   ├── analysis/            # Weight analysis tools
│   │   ├── weight_visualizer.py   # Distribution visualization
│   │   ├── sparsity_analyzer.py   # Sparsity pattern analysis
│   │   └── outlier_detector.py    # Outlier detection
│   ├── reports/             # Sample reports and templates
│   └── QUANTIZATION_GUIDE.md # Comprehensive quantization guide
├── pdk/                     # SkyWater PDK setup
├── synthesis/               # OpenLane synthesis scripts
├── pcb/                     # PCB design files
├── firmware/                # Card firmware
├── drivers/                 # Linux kernel driver
├── sdk/                     # Python SDK
├── fpga/                    # FPGA prototype files
└── tools/                   # Utility scripts
```

## Dependencies

### SkyWater SKY130 PDK

The open-source process design kit for 130nm CMOS fabrication.

```bash
# Clone the PDK (large download, ~7GB)
git clone https://github.com/google/skywater-pdk.git pdk/skywater-pdk

# Or add as submodule
git submodule add https://github.com/google/skywater-pdk.git pdk/skywater-pdk
```

**Documentation:** https://skywater-pdk.readthedocs.io/

### SmolVLM-256M Model

The 246M parameter vision-language model from Hugging Face.

```bash
# Install dependencies
pip install transformers torch pillow

# Download model
python -c "
from transformers import AutoProcessor, AutoModelForVision2Seq
processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct')
model = AutoModelForVision2Seq.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct')
processor.save_pretrained('model/smolvlm-256m')
model.save_pretrained('model/smolvlm-256m')
"
```

**Model Card:** https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct

### OpenLane (Synthesis)

RTL-to-GDSII flow using open-source tools.

```bash
# Install via Docker (recommended)
docker pull efabless/openlane:latest

# Or native installation
# See: https://openlane.readthedocs.io/en/latest/getting_started/installation.html
```

## Quick Start

### 🎮 Try the Interactive Demo

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

### 4. Model Quantization Workflow

The complete model conversion pipeline from FP32 to ternary weights:

```bash
# Step 1: Analyze model architecture and weights
python model/conversion/analyze_model.py --model HuggingFaceTB/SmolVLM-256M-Instruct

# Step 2: Run sensitivity analysis to identify critical layers
python model/conversion/sensitivity_analysis.py --model HuggingFaceTB/SmolVLM-256M-Instruct

# Step 3: Quantize to ternary weights
python model/conversion/quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --alpha 0.7 --mode per_tensor --export --output ./model/weights/quantized

# Step 4: Validate quantization quality
python model/conversion/validate_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --quantized ./model/weights/quantized --detailed

# Step 5: Run accuracy benchmarks
python model/validation/benchmark_suite.py --model HuggingFaceTB/SmolVLM-256M-Instruct --all
```

**For optimal accuracy**, use gradient-based alpha optimization:

```bash
python model/conversion/quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --optimize-alpha --export --output ./model/weights/optimized
```

See [QUANTIZATION_GUIDE.md](model/QUANTIZATION_GUIDE.md) for detailed instructions.

### 5. Run Tests

```bash
# E2E Pipeline Test (recommended first test)
python test_e2e_pipeline.py

# RTL simulation with cocotb
cd rtl/tb && make sim

# Python tests
pytest tests/
```

📖 **See [E2E Simulation Guide](docs/E2E_SIMULATION_GUIDE.md)** for detailed instructions on running the full simulation pipeline.

## Architecture Overview

📖 **Deep Dives:**
- [Attention Mechanism](docs/architecture/ATTENTION_MECHANISM.md) — How SiLens implements hardware-optimized attention with ternary weights, KV caching, RoPE, and approximate softmax
- [Full Architecture](docs/architecture/ARCHITECTURE.md) — Complete system design
- [E2E Simulation Guide](docs/E2E_SIMULATION_GUIDE.md) — How to run the RTL simulation pipeline

### Model: SmolVLM-256M

| Component | Parameters | Function |
|-----------|------------|----------|
| SigLIP-B/16 | 93M | Vision encoder (image → tokens) |
| Projector | 18M | Maps vision to language space |
| SmolLM2-135M | 135M | Language model (tokens → text) |
| **Total** | **246M** | |

### Hardware Target

| Specification | Value |
|---------------|-------|
| Process | SkyWater SKY130 (130nm) |
| Die size | ~800mm² |
| Package | BGA-625 |
| Interface | PCIe 3.0 x4 |
| Power | 25W TDP |
| Clock | 100-200 MHz |

### Hardwired Weight Encoding

```
Traditional: weights[i] stored in SRAM → read via memory bus → compute
SiLens:      weights[i] encoded as VDD/GND connection → instant compute

For 1-bit weights (ternary: -1, 0, +1):
  +1 → Metal trace to VDD
  -1 → Metal trace to GND
   0 → No connection (implicit zero)
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas Needing Help

- [ ] RTL design for transformer blocks
- [x] Weight quantization and validation ✓
- [ ] FPGA prototyping
- [ ] PCB design review
- [ ] Driver development
- [ ] Documentation

## Model Conversion Tools

The `model/` directory contains comprehensive tools for quantizing SmolVLM-256M:

### Quantization Pipeline

| Tool | Purpose |
|------|---------|
| `analyze_model.py` | Architecture analysis, weight statistics |
| `extract_weights.py` | Extract and organize weights |
| `quantize_ternary.py` | Ternary quantization with multiple modes |
| `calibration.py` | Calibration-aware quantization |
| `mixed_precision.py` | Keep critical layers at higher precision |
| `sensitivity_analysis.py` | Layer-by-layer sensitivity scoring |

### Validation Tools

| Tool | Purpose |
|------|---------|
| `validate_quantization.py` | Layer-by-layer quality validation |
| `semantic_equivalence_test.py` | **Verify original vs quantized model equivalence** |
| `benchmark_suite.py` | VQA, TextVQA, captioning benchmarks |
| `perplexity_test.py` | Language model perplexity measurement |
| `visual_qa_test.py` | Visual QA accuracy testing |
| `compare_outputs.py` | Side-by-side output comparison |

#### Semantic Equivalence Testing

Verify that quantized models produce semantically equivalent results:

```bash
# Run with default settings (normal tolerance)
python model/validation/semantic_equivalence_test.py

# Strict tolerance (for production-critical applications)
python model/validation/semantic_equivalence_test.py --tolerance strict

# Relaxed tolerance (for experimentation)
python model/validation/semantic_equivalence_test.py --tolerance relaxed --alpha 0.6

# Export detailed report
python model/validation/semantic_equivalence_test.py --output report.json
```

**Tolerance Levels:**
| Level | Weight Cosine | Token Match | Use Case |
|-------|---------------|-------------|----------|
| strict | > 0.95 | > 90% | Production deployment |
| normal | > 0.90 | > 80% | Development (default) |
| relaxed | > 0.80 | > 70% | Experimentation |

### Analysis Tools

| Tool | Purpose |
|------|---------|
| `weight_visualizer.py` | Distribution plots with matplotlib |
| `sparsity_analyzer.py` | Sparsity patterns and structured sparsity |
| `outlier_detector.py` | Identify and handle outlier weights |

### Expected Results

With default settings (α=0.7, per_tensor quantization):

| Metric | Original | Quantized | Change |
|--------|----------|-----------|--------|
| Memory | 1024 MB | 64 MB | 16x reduction |
| VQA Accuracy | ~71% | ~67% | ~4% drop |
| Perplexity | ~15 | ~17 | ~13% increase |
| Cosine Similarity | - | 0.92 | - |

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

### Third-Party Licenses

| Component | License | Source |
|-----------|---------|--------|
| SmolVLM-256M | Apache 2.0 | Hugging Face |
| SkyWater SKY130 | Apache 2.0 | Google/SkyWater |
| OpenLane | Apache 2.0 | Efabless |

## Acknowledgments

- **Hugging Face** for SmolVLM and the open model ecosystem
- **Google & SkyWater Technology** for the open PDK
- **Efabless** for OpenLane and open-source EDA tools
- The broader open-source hardware community

## Contact

- **Website:** [Coming soon]
- **Discord:** [Coming soon]
- **Email:** hello@silens.ai

---

*SiLens is an independent project and is not affiliated with Hugging Face, Google, SkyWater Technology, or any FPGA vendor.*
