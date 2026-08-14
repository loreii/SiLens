# SiLens Hierarchical Synthesis Strategy

> **Approach**: Bottom-up chiplet-style synthesis  
> **Goal**: Build DRC-clean hard macros, then integrate  
> **Process**: SkyWater SKY130

---

## Philosophy

Instead of trying to synthesize 800mm² in one shot (guaranteed failure), we:

1. **Identify reusable building blocks** (the "chiplets")
2. **Harden each block independently** (DRC clean, timing closed)
3. **Integrate as macros** in the final floorplan
4. **Iterate at each level** before moving up

This mirrors how real large chips are built - hierarchical hardening with proven IP blocks.

---

## Block Hierarchy - Current Status

```
Level 4: silens_soc_top (800mm²)
         ├── Integration of Level 3 macros
         └── RTL & Config: ✅ COMPLETE

Level 3: Subsystems (50-400mm² each) ✅ ALL COMPLETE
         ├── vision_subsystem (250mm²) = 12× transformer_block_vision
         ├── llm_subsystem (400mm²) = 30× transformer_block_llm  
         ├── memory_subsystem (50mm²) = ddr3_phy + controller + AXI arbiter
         └── io_subsystem (30mm²) = host_if + spi + gpio + interrupt

Level 2: Functional Blocks (5-20mm² each) ✅ ALL COMPLETE
         ├── transformer_block_vision (~20mm²)
         ├── transformer_block_llm (~13mm²)
         ├── projector_block (~10mm²)
         └── embedding_block (~15mm²)

Level 1: Compute Primitives (0.5-3mm² each) ✅ ALL COMPLETE
         ├── ternary_mac_array_64 (~1mm²)
         ├── rms_norm_block (~0.5mm²)
         ├── layer_norm_block (~0.5mm²)
         ├── softmax_unit (~0.5mm²)
         ├── silu_unit (~0.3mm²)
         ├── attention_head (~2mm²)
         └── mlp_block (~3mm²)

Level 0: Standard Cells
         └── SKY130 HD library
```

---

## Level 1: Compute Primitives

These are the smallest hardened blocks. Each should be:
- DRC clean
- Timing closed at 100MHz
- Characterized for power
- Reusable across vision and LLM

### 1.1 Ternary MAC Array (64-wide SIMD)

```
Module: ternary_mac_array_64
Size: ~1mm² (1000µm × 1000µm)
Inputs: 64 × 8-bit activations, 64 × 2-bit weights
Output: 32-bit accumulator
Latency: 1 cycle
```

**Synthesis target:**
```json
{
  "DESIGN_NAME": "ternary_mac_array_64",
  "CLOCK_PERIOD": 10.0,
  "DIE_AREA": "0 0 1000 1000",
  "PL_TARGET_DENSITY": 0.5
}
```

**Reuse**: This block is instantiated ~2000× across the full SoC.

### 1.2 RMS Normalization (576-dim)

```
Module: rms_norm_576
Size: ~0.5mm² (700µm × 700µm)
Inputs: 576 × 8-bit vector
Output: 576 × 8-bit normalized vector
Latency: ~50 cycles
```

**Reuse**: 60× in LLM (2 per layer), 24× in vision (2 per layer)

### 1.3 Softmax Unit

```
Module: softmax_approx
Size: ~0.5mm² (700µm × 700µm)
Inputs: Attention scores (sequence × heads)
Output: Attention weights
Latency: Variable (seq_len dependent)
```

**Reuse**: 30× in LLM, 12× in vision

### 1.4 Single Attention Head

```
Module: attention_head
Size: ~2mm² (1400µm × 1400µm)
Contains: Q/K/V projections, score compute, output
Params: head_dim=64, uses ternary_mac_array_64
```

**Reuse**: 9× per LLM layer (270 total), 12× per vision layer (144 total)

### 1.5 MLP Block

```
Module: mlp_block
Size: ~3mm² (1700µm × 1700µm)
Contains: Gate projection, Up projection, SiLU, Down projection
Architecture: 576 → 1536 → 576 (LLM) or 768 → 3072 → 768 (Vision)
```

**Reuse**: 30× in LLM, 12× in vision (different dimensions)

---

## Level 2: Functional Blocks

Compose Level 1 primitives into larger blocks.

### 2.1 Transformer Block (LLM variant)

```
Module: transformer_block_llm
Size: ~13mm² (3600µm × 3600µm)
Contains:
  - rms_norm_576 × 2
  - attention_head × 9
  - mlp_block × 1
  - Residual connections
```

**Critical path**: Attention → MLP → Output

### 2.2 Transformer Block (Vision variant)

```
Module: transformer_block_vision
Size: ~20mm² (4500µm × 4500µm)
Contains:
  - layer_norm_768 × 2
  - attention_head × 12
  - mlp_block × 1 (wider: 3072)
```

### 2.3 Projector Block

```
Module: projector_block
Size: ~10mm² (3200µm × 3200µm)
Contains:
  - ternary_mac_array_64 × many
  - Linear 768 → 576
```

### 2.4 Embedding Block

```
Module: embedding_block
Size: ~15mm² (3900µm × 3900µm)
Contains:
  - Token embedding ROM (49152 × 576)
  - Position embedding ROM (2048 × 576)
```

### 2.5 DDR3 PHY

```
Module: ddr3_phy
Size: ~15mm² (3900µm × 3900µm)
Contains:
  - IO buffers (LVDS-style for DQS)
  - DLL for clock alignment
  - Read/write leveling logic
```

**Note**: This is the trickiest block - analog/mixed-signal. May need custom layout.

---

## Level 3: Subsystems

### 3.1 Vision Subsystem

```
Module: vision_subsystem
Size: ~250mm² (15800µm × 15800µm)
Contains:
  - Patch embedding
  - transformer_block_vision × 12
  - Final layer norm
  - Output buffer
```

**Integration approach:**
- Place transformer blocks in 3×4 grid
- Clock tree per quadrant
- Shared weight buses between layers

### 3.2 LLM Subsystem

```
Module: llm_subsystem  
Size: ~400mm² (20000µm × 20000µm)
Contains:
  - transformer_block_llm × 30
  - embedding_block
  - LM head
  - KV cache interface
```

**Integration approach:**
- Place transformer blocks in 5×6 grid
- Pipelined token flow
- Shared memory bus for KV cache

### 3.3 Memory Subsystem

```
Module: memory_subsystem
Size: ~50mm² (7100µm × 7100µm)
Contains:
  - ddr3_phy
  - ddr3_controller
  - AXI arbiter
  - Refresh logic
```

### 3.4 IO Subsystem

```
Module: io_subsystem
Size: ~30mm² (5500µm × 5500µm)
Contains:
  - host_interface
  - spi_slave
  - gpio_controller
  - Interrupt controller
```

---

## Level 4: Top Integration

```
Module: silens_soc
Size: ~800mm² (26000µm × 30770µm)
```

**Floorplan:**

```
┌─────────────────────────────────────────────────────────────┐
│                    DDR3 IO (North)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌───────────────────┐    ┌───────────────────────────┐  │
│    │                   │    │                           │  │
│    │  VISION SUBSYS    │    │                           │  │
│    │    (250mm²)       │    │      LLM SUBSYSTEM        │  │
│    │                   │    │        (400mm²)           │  │
│    │  [12 xformers]    │    │                           │  │
│    │                   │    │      [30 xformers]        │  │
│    └───────────────────┘    │                           │  │
│                             │                           │  │
│    ┌───────────────────┐    │                           │  │
│    │   PROJECTOR       │    │                           │  │
│    │    (10mm²)        │    │                           │  │
│    └───────────────────┘    └───────────────────────────┘  │
│                                                             │
│    ┌───────────────────┐    ┌───────────────────────────┐  │
│    │   MEMORY SUBSYS   │    │     IO SUBSYSTEM          │  │
│    │    (50mm²)        │    │       (30mm²)             │  │
│    │  [DDR3 PHY+Ctrl]  │    │  [Host IF, SPI, GPIO]     │  │
│    └───────────────────┘    └───────────────────────────┘  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│              Clock/Power/Debug (South)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                         Host IO (East)
```

---

## Synthesis Order & Dependencies

```
Phase 1: Level 1 Primitives (Week 1-2)
├── [1] ternary_mac_array_64  (no deps)
├── [2] rms_norm_576          (no deps)
├── [3] layer_norm_768        (no deps)
├── [4] softmax_approx        (no deps)
└── [5] silu_unit             (no deps)

Phase 2: Level 2 Blocks (Week 3-4)
├── [6] attention_head        (deps: 1, 4)
├── [7] mlp_block_llm         (deps: 1, 5)
├── [8] mlp_block_vision      (deps: 1, 5)
├── [9] projector_block       (deps: 1)
└── [10] embedding_block      (no deps, ROM)

Phase 3: Level 2 Composed (Week 5-6)
├── [11] transformer_block_llm    (deps: 2, 6, 7)
├── [12] transformer_block_vision (deps: 3, 6, 8)
└── [13] ddr3_phy                 (custom, parallel track)

Phase 4: Level 3 Subsystems (Week 7-10)
├── [14] vision_subsystem     (deps: 12)
├── [15] llm_subsystem        (deps: 10, 11)
├── [16] memory_subsystem     (deps: 13)
└── [17] io_subsystem         (no macro deps)

Phase 5: Top Integration (Week 11-14)
└── [18] silens_soc           (deps: 14, 15, 16, 17)
```

---

## OpenLane Configuration per Block

### Template for Level 1 blocks

```json
{
  "DESIGN_NAME": "BLOCK_NAME",
  "CLOCK_PERIOD": 10.0,
  "FP_SIZING": "absolute",
  "DIE_AREA": "0 0 WIDTH HEIGHT",
  "PL_TARGET_DENSITY": 0.55,
  
  "//": "Harden as macro",
  "DESIGN_IS_CORE": false,
  
  "//": "Export for reuse",  
  "RUN_KLAYOUT_XOR": true,
  "RUN_MAGIC_DRC": true,
  "GENERATE_FINAL_SUMMARY_REPORT": true
}
```

### Template for Level 3+ integration

```json
{
  "DESIGN_NAME": "SUBSYSTEM_NAME",
  "CLOCK_PERIOD": 10.0,
  "FP_SIZING": "absolute",
  "DIE_AREA": "0 0 WIDTH HEIGHT",
  "PL_TARGET_DENSITY": 0.35,
  
  "//": "Use hardened macros",
  "MACRO_PLACEMENT_CFG": "macro_placement.cfg",
  "EXTRA_LEFS": ["block1.lef", "block2.lef"],
  "EXTRA_GDS_FILES": ["block1.gds", "block2.gds"],
  
  "//": "Macro-aware routing",
  "GLB_RT_OBS": "macro_obs.tcl"
}
```

---

## DRC Strategy per Level

### Level 1: Must be perfect
- Zero DRC violations
- Full antenna diode insertion
- Metal density balanced
- Timing margin: +20% (0.8ns slack minimum)

### Level 2: Must be clean
- Zero DRC after macro integration
- Clean interfaces between macros
- Timing margin: +10%

### Level 3: Iterative cleanup
- Initial run may have violations
- Focus on macro placement first
- Fix routing DRC iteratively
- Timing closure is primary goal

### Level 4: Integration focus
- Macro-to-macro routing only
- Power grid is critical
- Clock tree spans full die
- Accept longer iteration cycles

---

## Reuse Matrix

| Block | Vision Uses | LLM Uses | Total Instances |
|-------|-------------|----------|-----------------|
| ternary_mac_array_64 | ~500 | ~1500 | ~2000 |
| rms_norm_576 | 0 | 60 | 60 |
| layer_norm_768 | 24 | 0 | 24 |
| softmax_approx | 12 | 30 | 42 |
| attention_head (64-dim) | 144 | 270 | 414 |
| mlp_block_llm | 0 | 30 | 30 |
| mlp_block_vision | 12 | 0 | 12 |
| transformer_block_llm | 0 | 30 | 30 |
| transformer_block_vision | 12 | 0 | 12 |

---

## Risk Mitigation

### Risk: Block doesn't meet timing
**Mitigation**: 
- Start with 80MHz target, relax to 100MHz after closure
- Pipeline critical paths
- Accept lower clock for first tapeout

### Risk: Macro integration routing congestion
**Mitigation**:
- Leave 40% whitespace in floorplan
- Define routing channels between macros
- Use multiple metal layers for different signal types

### Risk: Power grid IR drop
**Mitigation**:
- Design PDN at subsystem level first
- Use M4/M5 power straps in macros
- Leave top metals (M6 if available) for global power

### Risk: Clock tree too large
**Mitigation**:
- Hierarchical clock distribution
- Local clock buffers in each macro
- Consider multiple clock domains

---

## File Organization

```
openlane/
├── level1/
│   ├── ternary_mac_array_64/
│   │   ├── config.json
│   │   ├── pin_order.cfg
│   │   └── src/ → symlink to rtl/
│   ├── rms_norm_576/
│   └── ...
├── level2/
│   ├── transformer_block_llm/
│   ├── transformer_block_vision/
│   └── ...
├── level3/
│   ├── vision_subsystem/
│   ├── llm_subsystem/
│   └── ...
└── level4/
    └── silens_soc/
```

---

## Next Steps

1. **Create Level 1 configs** - Start with `ternary_mac_array_64`
2. **Run first synthesis** - Get baseline metrics
3. **Iterate to DRC clean** - Fix violations
4. **Characterize** - Extract timing/power
5. **Move to Level 2** - Integrate proven blocks

---

*This is a multi-month effort. FPGA prototyping runs in parallel to validate RTL correctness.*
