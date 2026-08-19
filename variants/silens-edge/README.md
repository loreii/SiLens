# SiLens Edge - Ultra-Fast Vision Classifier

> **Status:** In Development  
> **Target:** 50mm² die, <1ms latency, 1000+ FPS

SiLens Edge is a compact, manufacturable variant of the SiLens architecture designed for **edge deployment** and **industrial applications**.

---

## Key Differences from SiLens VLM

| Specification | SiLens VLM | SiLens Edge |
|---------------|------------|-------------|
| **Die Size** | 800mm² | 50mm² |
| **Parameters** | 246M | 20M |
| **Power** | 25W | 3W |
| **Latency** | <5ms | <1ms |
| **Throughput** | 200 FPS | 1000+ FPS |
| **Output** | Text generation | Single token/class |
| **Package** | BGA-900 | QFN-48 |
| **Cost** | ~$500 | ~$50 |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│            SiLens Edge (50mm²)                  │
│                                                 │
│  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  NanoViT        │  │  TinyLLM            │  │
│  │  Vision Encoder │──│  Classifier Head    │  │
│  │  (12M params)   │  │  (7M params)        │  │
│  │  6 layers       │  │  6 layers           │  │
│  │  192-dim        │  │  128-dim            │  │
│  └─────────────────┘  └─────────────────────┘  │
│           │                    │                │
│           └────────┬───────────┘                │
│                    ▼                            │
│           ┌───────────────┐                     │
│           │ Classification │                    │
│           │    Output      │                    │
│           │ (1000 classes) │                    │
│           └───────────────┘                     │
│                                                 │
│  ┌─────────────┐  ┌────────────┐               │
│  │ SPI/I2C     │  │  GPIO      │               │
│  │ Interface   │  │  Triggers  │               │
│  └─────────────┘  └────────────┘               │
└─────────────────────────────────────────────────┘
```

---

## Use Cases

### Industrial Quality Control
```
Input:  [Image of gear on assembly line]
Output: "DEFECT" or "PASS"
Latency: 0.8ms
```

### Safety Monitoring
```
Input:  [Security camera frame]
Output: "PERSON_DETECTED" or "CLEAR"
Latency: 0.5ms
```

### Drone Navigation
```
Input:  [Forward camera view]
Output: "OBSTACLE" / "CLEAR" / "LANDING_ZONE"
Latency: 1ms
```

### Robotics
```
Input:  [Gripper camera view]
Output: "OBJECT_PRESENT" / "ALIGNED" / "MISALIGNED"
Latency: 0.3ms
```

---

## Why This Matters

The full SiLens VLM (800mm²) is a **moonshot** - technically challenging and expensive to manufacture. SiLens Edge:

1. **Validates the hardwired approach** at smaller scale
2. **Ships first** - proving the team can deliver
3. **Lower Kickstarter goal** - $50K vs $500K
4. **Real market demand** - industrial edge AI is a $10B+ market
5. **Learning opportunity** - OpenLane, tape-out, PCB at manageable scale

---

## Development Status

| Component | Status |
|-----------|--------|
| Architecture spec | 🟡 In progress |
| Level 1-2 primitives | ✅ Shared with VLM |
| Level 3 subsystems | 🔴 Not started |
| Level 4 top integration | 🔴 Not started |
| Model selection | 🟡 Evaluating options |
| OpenLane synthesis | 🔴 Not started |
| FPGA prototype | 🔴 Not started |

---

## Building

```bash
# From SiLens root
make VARIANT=silens-edge level3
make VARIANT=silens-edge level4
make VARIANT=silens-edge all
```

---

## File Structure

```
variants/silens-edge/
├── config.json          # Variant configuration
├── README.md            # This file
├── openlane/
│   ├── level3/          # Edge-specific subsystems
│   └── level4/          # Edge top integration
├── rtl/
│   └── silens_edge_soc.v
└── docs/
    └── kickstarter/     # Edge-specific campaign
```

---

## Contributing

See the main [CONTRIBUTING.md](../../CONTRIBUTING.md). Focus areas for Edge:

1. Small vision encoder evaluation (MobileViT, EfficientViT, etc.)
2. Tiny LLM classifier design
3. QFN/BGA package pinout
4. Low-power design techniques
