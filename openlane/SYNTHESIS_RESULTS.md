# SiLens OpenLane Synthesis Results

> **Last Updated:** August 19, 2026  
> **OpenLane Version:** v1.0.2  
> **PDK:** SkyWater SKY130A  
> **Target Clock:** 71 MHz (14ns period)

This document tracks synthesis results for all hierarchical blocks.

---

## Summary

| Level | Total | Passed | Failed | Pending |
|-------|-------|--------|--------|---------|
| Level 1 | 7 | 2 | 2 | 3 |
| Level 2 | 4 | 0 | 0 | 4 |
| Level 3 (VLM) | 4 | 0 | 0 | 4 |
| Level 3 (Edge) | 4 | 0 | 0 | 4 |
| Level 4 (VLM) | 1 | 0 | 0 | 1 |
| Level 4 (Edge) | 1 | 0 | 0 | 1 |

---

## Level 1: Compute Primitives

These are the fundamental building blocks, reused throughout the design.

### ternary_mac_array_64

| Metric | Value |
|--------|-------|
| **Status** | ✅ PASS |
| **Date** | 2026-08-19 |
| **Die Area** | 1.0 mm² |
| **Cell Count** | 5,306 |
| **Utilization** | 5.78% |
| **Wire Length** | 613,465 µm |
| **Runtime** | 10m 28s |
| **DRC Violations** | 0 |
| **LVS Errors** | 0 |
| **Hold Violations** | 0 |
| **Setup WNS** | -3.42 ns @ 10ns clock |
| **Suggested Clock** | 13.53 ns (~74 MHz) |

**Notes:**
- First successful synthesis run
- Clock period relaxed to 14ns for clean timing
- Ready for reuse in Level 2 blocks

**Output Files:**
- GDS: `level1/ternary_mac_array_64/runs/run1/results/final/gds/`
- LEF: `level1/ternary_mac_array_64/runs/run1/results/final/lef/`
- Timing LIB: `level1/ternary_mac_array_64/runs/run1/results/final/lib/`

---

### rms_norm_block

| Metric | Value |
|--------|-------|
| **Status** | ⚠️ TIMEOUT (Synthesis) |
| **Date** | 2026-08-19 |
| **Issue** | 576-dim too large, ABC optimization >30min |

**Notes:**
- Design is lint-clean after fixes
- Synthesis taking too long for 576-dimension
- Consider reducing DIM to 64 or 128 for faster iteration
- Or increase placement density for smaller area

---

### layer_norm_block

| Metric | Value |
|--------|-------|
| **Status** | 🔧 RTL FIXED (Ready to synthesize) |
| **Date** | 2026-08-19 |

**Notes:**
- Fixed Verilator linting issues (generate blocks, automatic function)
- 768-dim version - may take long to synthesize like rms_norm

---

### softmax_unit

| Metric | Value |
|--------|-------|
| **Status** | ⚠️ PASS (Timing Violations) |
| **Date** | 2026-08-19 |
| **Die Area** | 0.49 mm² |
| **Cell Count** | 5,447 |
| **Utilization** | 14.48% |
| **Wire Length** | 244,975 µm |
| **Runtime** | 5m 22s |
| **DRC Violations** | 0 |
| **LVS Errors** | 0 |
| **Setup WNS** | -14.77 ns @ 10ns clock |
| **Suggested Clock** | 25.25 ns (~40 MHz) |

**Notes:**
- Design physically correct (DRC/LVS clean)
- Severe timing violations - needs clock relaxation or pipelining
- Exponential approximation is compute-heavy

---

### silu_unit

| Metric | Value |
|--------|-------|
| **Status** | ❌ FAIL (RTL Error) |
| **Date** | 2026-08-19 |
| **Error** | Multiple conflicting drivers for register 'i' |

**Notes:**
- RTL has multiple always blocks driving same signal
- Needs refactoring to fix driver conflicts

---

### attention_head

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |

---

### mlp_block

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |

---

## Level 2: Functional Blocks

### transformer_block_llm

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~13 mm² |

---

### transformer_block_vision

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~20 mm² |

---

### projector_block

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~10 mm² |

---

### embedding_block

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~15 mm² |

---

## Level 3: Subsystems (SiLens VLM)

### vision_subsystem

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~250 mm² |

---

### llm_subsystem

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~400 mm² |

---

### memory_subsystem

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~50 mm² |

---

### io_subsystem

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~30 mm² |

---

## Level 3: Subsystems (SiLens Edge)

### vision_nano

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~15 mm² |

---

### classifier_head

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~10 mm² |

---

### io_edge

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~5 mm² |

---

### sram_256kb

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | ~10 mm² |

---

## Level 4: Top Integration

### silens_soc (VLM - 800mm²)

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | 800 mm² |

---

### silens_edge_soc (Edge - 50mm²)

| Metric | Value |
|--------|-------|
| **Status** | ⏳ PENDING |
| **Target Area** | 50 mm² |

---

## Lessons Learned

### 2026-08-19: Initial Synthesis Runs

1. **Clock Period:** 100MHz (10ns) is too aggressive for Level 1 blocks on SKY130. 
   - Achieved: ~74 MHz for MAC array, ~40 MHz for softmax
   - Action: Relaxed to 14ns (71 MHz) for timing closure

2. **Verilator Linter:** OpenLane uses Verilator for linting which is stricter than iverilog.
   - Issue: Delayed assignments in for loops not supported
   - Fix: Use generate blocks for parallel register loading
   - Issue: Functions default to static
   - Fix: Use `function automatic` keyword

3. **Utilization:** MAC array at 5.78% utilization - can pack much denser.
   - Consider reducing die area or adding more compute units

4. **Large Dimension Blocks:** 576-dim and 768-dim blocks take >30 min in ABC.
   - Consider reducing dimension for iteration or using hierarchical synthesis

5. **RTL Quality:** Several blocks have multi-driver issues.
   - silu_unit needs refactoring for single-driver discipline
   
6. **Successful Synthesis:** ternary_mac_array_64 and softmax_unit prove the flow works.
   - DRC/LVS clean achievable
   - Timing can be fixed with clock relaxation or pipelining

---

## Environment

```
OpenLane: v1.0.2 (ff5509f65b17bfa4068d5336495ab1718987ff69)
Docker Image: ghcr.io/the-openroad-project/openlane:latest
PDK: sky130A (0fe599b2afb6708d281543108caf8310912f54af)
Host: macOS (Apple Silicon arm64)
```

---

## Commands Reference

```bash
# Run single block synthesis
cd ~/OpenLane
docker run --rm \
  -v $(pwd):/openlane \
  -v $HOME/pdk:/home/pdk \
  -e PDK_ROOT=/home/pdk \
  -e PDK=sky130A \
  ghcr.io/the-openroad-project/openlane:latest \
  ./flow.tcl -design <block_name> -tag run1

# Check synthesis status
cd /path/to/SiLens/openlane
make status

# View metrics
cat ~/OpenLane/designs/<block>/runs/run1/reports/metrics.csv
```
