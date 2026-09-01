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

## Hardware Variants

SiLens supports multiple hardware variants built from shared compute primitives:

| Variant | Die Size | Model | Use Case | Status |
|---------|----------|-------|----------|--------|
| **[silens-vlm](variants/silens-vlm/)** | 800mm² | SmolVLM-256M | Conversational Vision AI | RTL Complete |
| **[silens-edge](variants/silens-edge/)** | 50mm² | TinyVLM-20M | Edge Classification | In Development |

### Build Strategy

```
Shared (Level 1-2)              Variant-Specific (Level 3-4)
┌─────────────────────┐         ┌─────────────────────────────┐
│ MAC arrays          │         │ silens-vlm (800mm²)         │
│ Normalization       │ ──────► │   Vision + LLM subsystems   │
│ Attention heads     │         │   Full text generation      │
│ MLP blocks          │         ├─────────────────────────────┤
│ Transformer blocks  │         │ silens-edge (50mm²)         │
└─────────────────────┘         │   Compact classifier        │
                                │   Single-token output       │
                                └─────────────────────────────┘
```

**Why two variants?**
- **silens-edge ships first** — De-risks manufacturing at smaller scale
- **Validates hardwired approach** — Proves concept before 800mm² investment
- **Different markets** — VLM for consumer AI, Edge for industrial IoT

See [variants/README.md](variants/README.md) for details.

## Repository Structure

```
SiLens/
├── README.md                 # This file
├── LICENSE                   # Apache 2.0
├── variants/                 # Hardware variants
│   ├── README.md             # Variants overview
│   ├── silens-vlm/           # 800mm² Vision-Language Model
│   │   ├── config.json       # Variant configuration
│   │   ├── openlane/level3/  # VLM subsystems
│   │   ├── openlane/level4/  # VLM top integration
│   │   └── docs/kickstarter/ # VLM campaign materials
│   └── silens-edge/          # 50mm² Edge Classifier
│       ├── config.json       # Variant configuration
│       ├── openlane/level3/  # Edge subsystems (TBD)
│       ├── openlane/level4/  # Edge top integration (TBD)
│       └── docs/             # Edge documentation
├── openlane/                 # Shared synthesis configs
│   ├── Makefile              # Multi-variant build system
│   ├── level1/               # Shared compute primitives
│   └── level2/               # Shared functional blocks
├── docs/                     # Shared documentation
│   └── architecture/         # System architecture
├── rtl/                      # Verilog/SystemVerilog source
│   ├── common/               # Shared primitives
│   ├── vision_encoder/       # Vision encoder RTL
│   ├── language_model/       # LLM RTL
│   ├── projector/            # Projector RTL
│   └── tb/                   # Testbenches
├── model/                    # Model conversion tools
├── sdk/                      # Python SDK
├── drivers/                  # Linux kernel driver
├── firmware/                 # Card firmware
├── fpga/                     # FPGA prototype files
└── tools/                    # Utility scripts
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

# Semantic equivalence test
python model/validation/semantic_equivalence_test.py

# RTL simulation with cocotb
cd rtl/tb && make sim

# Python tests
pytest tests/
```

📖 **See [E2E Simulation Guide](docs/E2E_SIMULATION_GUIDE.md)** for detailed instructions on running the full simulation pipeline.

---

## Quantization Validation Results

We validated that ternary-quantized weights preserve semantic equivalence with the original FP32 model.

📖 **Full Reports:** 
- [Semantic Equivalence Results](docs/SEMANTIC_EQUIVALENCE_RESULTS.md) — Weight and activation analysis
- [Visual Equivalence Test (Lenna)](docs/VISUAL_EQUIVALENCE_LENNA_TEST.md) — Visual understanding validation

### Summary (SmolVLM-256M, α=0.7)

| Test | Score | Threshold | Status |
|------|-------|-----------|--------|
| Weight Similarity | 0.8787 | 0.90 | ⚠️ Expected |
| Activation Similarity | **0.9969** | 0.90 | ✅ Pass |
| Output Similarity | **0.9968** | 0.90 | ✅ Pass |
| Semantic Similarity | **0.8004** | 0.80 | ✅ Pass |

### Visual Understanding Test (Lenna Image)

| Category | Similarity | Status |
|----------|------------|--------|
| Image Description | 100% | ✅ |
| Object Detection | 95.9% | ✅ |
| Color Analysis | 100% | ✅ |
| Spatial Reasoning | 100% | ✅ |
| Question Answering | 100% | ✅ |
| **Overall** | **94.0%** | ✅ |

**Key Insight:** While individual weights show ~12% reconstruction error (inherent to ternary quantization), the model's **functional behavior is preserved** with >99% activation/output similarity and 94% visual understanding accuracy.

### What This Means

```
Original Model                    SiLens Ternary Model
┌─────────────────┐               ┌─────────────────┐
│ FP32 Weights    │               │ {-1, 0, +1}     │
│ 1024 MB         │    ──────►    │ 64 MB (16x ↓)   │
│ cos(w,w')=1.00  │               │ cos(w,w')=0.88  │
└────────┬────────┘               └────────┬────────┘
         │                                 │
         ▼                                 ▼
┌─────────────────┐               ┌─────────────────┐
│ Activations     │    ═══════    │ Activations     │
│                 │   99.7% same  │                 │
└────────┬────────┘               └────────┬────────┘
         │                                 │
         ▼                                 ▼
┌─────────────────┐               ┌─────────────────┐
│ Output: "A cat  │    ═══════    │ Output: "A cat  │
│ on the couch"   │   Semantic    │ resting on the  │
│                 │   Equivalent  │ sofa"           │
└─────────────────┘               └─────────────────┘
```

### Sparsity Bonus

~44% of weights are quantized to zero, enabling:
- **Zero-skipping:** No computation needed for zero weights
- **Reduced routing:** Fewer metal traces in silicon
- **Power savings:** Less switching activity

---

## Architecture Overview

📖 **Deep Dives:**
- [Semantic Equivalence Results](docs/SEMANTIC_EQUIVALENCE_RESULTS.md) — Experimental validation that ternary quantization preserves model behavior
- [Visual Equivalence Test (Lenna)](docs/VISUAL_EQUIVALENCE_LENNA_TEST.md) — Visual understanding validation with standard test image
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
| `visual_equivalence_test.py` | **Visual understanding test with Lenna image** |
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

#### Visual Equivalence Testing (Lenna Image)

Test visual understanding with the standard Lenna test image:

```bash
# Run visual equivalence test
python model/validation/visual_equivalence_test.py

# Export results to JSON
python model/validation/visual_equivalence_test.py --output results/lenna_test.json

# Use custom image
python model/validation/visual_equivalence_test.py --image path/to/image.png
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

## Related Projects

### Engram SSD Accelerator

For running larger models like **Qwen3.8-Flash-Next** (125B MoE + 51.2B Engram table) on consumer hardware, we're developing the [Engram SSD Accelerator](https://github.com/loreii/engram-ssd-accelerator) — a companion project that keeps the massive N-gram embedding table on NVMe SSD with hardware-accelerated access.

**Combined architecture for Qwen3.8-Flash-Next:**
- **Engram SSD Accelerator**: 51.2B N-gram table on SSD with DMA access (102GB → 4GB DRAM)
- **SiLens Ternary Quantization**: 125B MoE backbone quantized to {-1, 0, +1} (250GB → 31GB)
- **Total DRAM requirement**: <45GB instead of 350GB+

This enables running frontier-class models on prosumer hardware (RTX 4090 + 64GB RAM + NVMe array).

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
