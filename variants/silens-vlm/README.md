# SiLens VLM - Full Vision-Language Model Accelerator

> **Status:** RTL Complete, Physical Design In Progress  
> **Target:** 800mm² die, SmolVLM-256M hardwired

SiLens VLM is the flagship variant - a full **Vision-Language Model** accelerator with 246 million parameters hardwired into silicon.

---

## Specifications

| Specification | Value |
|---------------|-------|
| **Die Size** | 800mm² (26mm × 30.77mm) |
| **Process** | SkyWater SKY130 130nm |
| **Parameters** | 246M (ternary quantized) |
| **Model** | SmolVLM-256M-Instruct |
| **Power** | 25W TDP |
| **Latency** | <5ms per inference |
| **Throughput** | 200+ images/second |
| **Interface** | DDR3-1066, Parallel host bus |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SiLens VLM (800mm²)                      │
│                                                             │
│  ┌───────────────────┐    ┌───────────────────────────┐    │
│  │  VISION ENCODER   │    │      LANGUAGE MODEL       │    │
│  │    (SigLIP-B)     │    │      (SmolLM2-135M)       │    │
│  │    250mm²         │    │        400mm²             │    │
│  │   93M params      │───▶│      135M params          │    │
│  │   12 layers       │    │       30 layers           │    │
│  └───────────────────┘    └───────────────────────────┘    │
│           │                          │                      │
│           │         ┌────────────────┘                      │
│           │         │                                       │
│  ┌────────┴─────────┴────────┐  ┌─────────────────────────┐│
│  │   MEMORY SUBSYSTEM        │  │    IO SUBSYSTEM         ││
│  │   DDR3 + AXI Arbiter      │  │  Host IF + SPI + GPIO   ││
│  │        50mm²              │  │        30mm²            ││
│  └───────────────────────────┘  └─────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Hierarchical Build Status

| Level | Description | Blocks | Status |
|-------|-------------|--------|--------|
| Level 1 | Compute Primitives | 7 blocks | ✅ Complete |
| Level 2 | Functional Blocks | 4 blocks | ✅ Complete |
| Level 3 | Subsystems | 4 subsystems | ✅ Complete |
| Level 4 | Top Integration | 1 (800mm²) | ✅ Complete |

---

## Use Cases

- **Image Captioning**: "Describe this image in detail"
- **Visual QA**: "What color is the car in this photo?"
- **Scene Understanding**: "Is there a person near the door?"
- **Multi-turn Dialogue**: Follow-up questions about images

---

## Building

```bash
# From SiLens root
make VARIANT=silens-vlm level1    # Shared primitives
make VARIANT=silens-vlm level2    # Shared blocks
make VARIANT=silens-vlm level3    # VLM-specific subsystems
make VARIANT=silens-vlm level4    # 800mm² top integration
```

---

## File Structure

```
variants/silens-vlm/
├── config.json              # Variant configuration
├── README.md                # This file
├── openlane/
│   ├── level3/              # 800mm² subsystems
│   │   ├── vision_subsystem/
│   │   ├── llm_subsystem/
│   │   ├── memory_subsystem/
│   │   └── io_subsystem/
│   └── level4/
│       └── silens_soc/      # Top integration
├── rtl/
│   ├── silens_soc.v         # Top-level SoC
│   └── silens_vlm_core.v    # VLM processing core
└── docs/
    ├── kickstarter/         # VLM Kickstarter campaign
    └── business/            # Business plans
```

---

## Technical Challenges

This is the **moonshot** variant with significant physical design challenges:

1. **Routing Congestion**: 246M hardwired connections
2. **IR Drop**: 25W power delivery across 28mm span
3. **Clock Tree**: 100MHz distribution to 800mm²
4. **Yield**: Large die on mature process

See [SILENS_TECHNICAL_EVALUATION.md](../../docs/SILENS_TECHNICAL_EVALUATION.md) for detailed analysis.

---

## Why Build This?

- **Eliminate memory bottleneck** - Near-zero latency weight access
- **Power efficiency** - No DRAM fetch energy
- **Open-source silicon** - Fully reproducible hardware AI
- **Push boundaries** - Prove hardwired AI is viable

The Edge variant de-risks this by validating the approach at smaller scale first.
