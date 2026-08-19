# SiLens Variants

SiLens supports multiple hardware variants built from shared compute primitives.

---

## Available Variants

| Variant | Die Size | Model | Target Use Case | Status |
|---------|----------|-------|-----------------|--------|
| **silens-vlm** | 800mm² | SmolVLM-256M | Conversational Vision AI | RTL Complete |
| **silens-edge** | 50mm² | TinyVLM-20M | Edge Classification | In Development |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SHARED COMPONENTS                         │
│                                                                  │
│   openlane/level1/          openlane/level2/                    │
│   ├── ternary_mac           ├── transformer_block_llm           │
│   ├── rms_norm              ├── transformer_block_vision        │
│   ├── layer_norm            ├── projector_block                 │
│   ├── softmax               └── embedding_block                 │
│   ├── silu                                                      │
│   ├── attention_head        rtl/common/                         │
│   └── mlp_block             sdk/  drivers/  model/              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
┌───────────────────────┐           ┌───────────────────────┐
│   variants/silens-vlm │           │  variants/silens-edge │
│                       │           │                       │
│   800mm² VLM          │           │   50mm² Classifier    │
│   246M parameters     │           │   20M parameters      │
│   25W TDP             │           │   3W TDP              │
│                       │           │                       │
│   openlane/           │           │   openlane/           │
│   ├── level3/         │           │   ├── level3/         │
│   │   ├── vision_sub  │           │   │   ├── nano_vision │
│   │   ├── llm_sub     │           │   │   └── classifier  │
│   │   ├── memory_sub  │           │   └── level4/         │
│   │   └── io_sub      │           │       └── edge_soc    │
│   └── level4/         │           │                       │
│       └── silens_soc  │           │   docs/kickstarter/   │
│                       │           │                       │
│   docs/kickstarter/   │           │                       │
└───────────────────────┘           └───────────────────────┘
```

---

## Building Variants

```bash
# Build shared primitives (required for all variants)
cd openlane
make level1 level2

# Build the full VLM (800mm²)
make VARIANT=silens-vlm level3 level4

# Build the Edge classifier (50mm²)
make VARIANT=silens-edge level3 level4

# Check status
make VARIANT=silens-vlm status
make VARIANT=silens-edge status

# List all variants
make list-variants
```

---

## Creating a New Variant

1. Create directory structure:
```bash
mkdir -p variants/my-variant/{openlane/{level3,level4},rtl,docs}
```

2. Create `config.json` with variant parameters

3. Add Level 3 subsystems and Level 4 top integration

4. Build:
```bash
make VARIANT=my-variant all
```

---

## Variant Comparison

### silens-vlm (Moonshot)
- **Goal:** Full conversational vision-language AI
- **Risk:** High (routing congestion, power delivery, yield)
- **Timeline:** 12-18 months
- **Kickstarter:** $500K+ target
- **Output:** Multi-token text generation

### silens-edge (Practical)
- **Goal:** Ultra-fast edge vision classifier
- **Risk:** Low (proven scale, manufacturable)
- **Timeline:** 6-9 months
- **Kickstarter:** $50K target
- **Output:** Single classification token

---

## Strategy

1. **Build Edge first** - Validate hardwired approach at 50mm² scale
2. **Ship to backers** - Prove team can deliver hardware
3. **Learn from Edge** - OpenLane, tape-out, PCB, power delivery
4. **De-risk VLM** - Apply learnings to 800mm² design
5. **Launch VLM Kickstarter** - With Edge success as credibility
