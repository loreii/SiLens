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
| RTL design (Verilog) | 🔴 Not started |
| FPGA prototype | 🔴 Not started |
| Physical design | 🔴 Not started |
| PCB design | 🔴 Not started |
| Software/drivers | 🔴 Not started |

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
│   └── validation/          # Model accuracy validation
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

### 1. Clone Repository

```bash
git clone https://github.com/[your-org]/SiLens.git
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

### 4. Run Tests (when available)

```bash
# RTL simulation
make sim

# Python tests
pytest tests/
```

## Architecture Overview

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
- [ ] Weight quantization and validation
- [ ] FPGA prototyping
- [ ] PCB design review
- [ ] Driver development
- [ ] Documentation

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
