# SiLens Attention Mechanism

> **Hardware-optimized attention for ternary weight vision-language models**

This document explains how SiLens implements attention in custom silicon, optimized for **ternary weights** and **fixed-point arithmetic**.

## Overview

SiLens implements attention across four interconnected RTL modules:

| Module | File | Purpose |
|--------|------|---------|
| LLM Attention | `rtl/language_model/llm_attention.v` | Grouped-Query Attention for text generation |
| ViT Attention | `rtl/vision_encoder/vit_attention.v` | Multi-Head Self-Attention for vision |
| Softmax | `rtl/common/softmax_approx.v` | Hardware-friendly approximate softmax |
| KV Cache | `rtl/memory/kv_cache.v` | Key-Value cache for autoregressive decoding |

## Ternary Weight Projections

All Q/K/V/O projection matrices use **2-bit ternary encoding**, eliminating multipliers entirely:

```verilog
// Weight encoding
localparam W_ZERO = 2'b00;  // 0
localparam W_POS  = 2'b01;  // +1
localparam W_NEG  = 2'b10;  // -1

// Ternary MAC: add, subtract, or nothing
function signed [ACC_WIDTH-1:0] ternary_mac;
    input [ACT_WIDTH-1:0] activation;
    input [1:0] weight;
    begin
        case (weight)
            W_POS:   ternary_mac = +$signed({1'b0, activation});
            W_NEG:   ternary_mac = -$signed({1'b0, activation});
            default: ternary_mac = 0;
        endcase
    end
endfunction
```

**Benefits:**
- No DSP blocks or multipliers needed
- Simple add/subtract logic gates
- 16x weight compression (FP32 → 2-bit)
- Lower power consumption

## Two Attention Architectures

SiLens implements different attention variants optimized for each modality:

### Language Model: Grouped-Query Attention (GQA)

| Parameter | Value |
|-----------|-------|
| Heads | 9 |
| Dimension | 576 (64 per head) |
| Max Sequence | 8192 tokens |
| KV Heads | 9 (can share for efficiency) |
| Position Encoding | RoPE (Rotary) |

**Key features:**
- **KV Cache** for autoregressive generation
- **Grouped-Query** allows sharing KV across multiple query heads
- **RoPE** for relative position understanding

### Vision Transformer: Multi-Head Self-Attention

| Parameter | Value |
|-----------|-------|
| Heads | 12 |
| Dimension | 768 (64 per head) |
| Sequence Length | 576 patches |
| Position Encoding | None (patch position implicit) |

**Key features:**
- Processes entire image at once
- No KV cache needed (not autoregressive)
- Scaled dot-product attention

## KV Cache for Autoregressive Generation

The `kv_cache.v` module stores previous Key and Value vectors, so each new token only computes its own projections:

```
Token 1: Q₁ attends to K₁, V₁
Token 2: Q₂ attends to K₁, K₂, V₁, V₂  (K₁, V₁ cached)
Token 3: Q₃ attends to all cached K, V
...
Token N: Qₙ attends to all N cached K, V vectors
```

### Memory Organization

```verilog
// Per-head BRAM memories
genvar h;
generate
    for (h = 0; h < NUM_HEADS; h = h + 1) begin : gen_head_cache
        (* ram_style = "block" *)
        reg [HEAD_DIM*ACT_WIDTH-1:0] k_mem [0:MAX_SEQ_LEN-1];
        
        (* ram_style = "block" *)
        reg [HEAD_DIM*ACT_WIDTH-1:0] v_mem [0:MAX_SEQ_LEN-1];
    end
endgenerate
```

### Cache Sizing

| Configuration | Memory per Layer |
|---------------|------------------|
| 9 heads × 64 dims × 8 bits | 576 bytes per position |
| 2048 on-chip positions | ~1.15 MB per layer |
| 30 layers (full model) | ~34.5 MB on-chip cache |

For longer sequences (8K+), external memory via AXI interface is used.

## RoPE (Rotary Position Embeddings)

The LLM attention applies RoPE to encode position information without learned embeddings:

```verilog
// RoPE formula: x' = x * cos(θ) + rotate(x) * sin(θ)
function signed [ACT_WIDTH-1:0] apply_rope;
    input signed [ACT_WIDTH-1:0] x_even;
    input signed [ACT_WIDTH-1:0] x_odd;
    input signed [ACT_WIDTH-1:0] cos_val;
    input signed [ACT_WIDTH-1:0] sin_val;
    input integer is_odd;
    reg signed [ACC_WIDTH-1:0] result;
    begin
        if (is_odd) begin
            // Odd positions: x*cos + x_prev*sin
            result = (x_odd * cos_val + x_even * sin_val) >>> FRAC_BITS;
        end else begin
            // Even positions: x*cos - x_next*sin
            result = (x_even * cos_val - x_odd * sin_val) >>> FRAC_BITS;
        end
        apply_rope = saturate(result);
    end
endfunction
```

**How it works:**
- Precomputed `cos(θ)` and `sin(θ)` tables indexed by position
- Rotation applied to Q and K vectors before attention
- Enables relative position understanding (token distance matters, not absolute position)

## Hardware-Friendly Softmax

The `softmax_approx.v` module avoids expensive exponential computation using **piece-wise linear approximation**:

### Approximation Segments

```
softmax(xᵢ) = exp(xᵢ) / Σexp(xⱼ)

For x in [-8, 0] (after subtracting max for stability):
┌─────────┬──────────────────────────────────────┐
│ Range   │ Approximation                        │
├─────────┼──────────────────────────────────────┤
│ [-8,-4] │ exp(x) ≈ 0 (negligible)              │
│ [-4,-2] │ exp(x) ≈ 0.018 + 0.058*(x+4)         │
│ [-2,-1] │ exp(x) ≈ 0.135 + 0.233*(x+2)         │
│ [-1, 0] │ exp(x) ≈ 0.368 + 0.632*(x+1)         │
└─────────┴──────────────────────────────────────┘
```

### Processing Pipeline

```verilog
// FSM states
localparam STATE_FIND_MAX    = 3'd1;  // Numerical stability
localparam STATE_COMPUTE_EXP = 3'd2;  // Piece-wise linear exp
localparam STATE_SUM_EXP     = 3'd3;  // Accumulate denominator
localparam STATE_NORMALIZE   = 3'd4;  // Multiply by reciprocal
```

1. **Find Max** — Parallel reduction to find maximum value
2. **Subtract Max** — Numerical stability (prevents overflow)
3. **Compute exp()** — Piece-wise linear approximation
4. **Sum exp values** — Accumulate denominator
5. **Normalize** — Multiply by reciprocal (LUT-based division)

## Attention Score Computation

Scaled dot-product with bit-shift instead of division:

```verilog
// sqrt(HEAD_DIM) = sqrt(64) = 8
// Division by 8 = right shift by 3 bits
localparam SCALE_SHIFT = 3;

// Attention score computation
score_accum = 0;
for (i = 0; i < HEAD_DIM; i = i + 1) begin
    score_accum = score_accum + 
        $signed(q_buf[head_idx * HEAD_DIM + i]) *
        $signed(k_cache[cache_idx][head_idx * HEAD_DIM + i]);
end

// Scale by 1/sqrt(d_k)
attn_scores[cache_idx] <= saturate(score_accum >>> SCALE_SHIFT);
```

## Complete Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Input Token (576/768 dims, 8-bit activations)              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Ternary Projections: Q = X·Wq, K = X·Wk, V = X·Wv          │
│  (add/subtract only — no multipliers needed)                │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  RoPE: Q, K rotated by position (LLM only)                  │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Cache K, V (LLM) or buffer entire sequence (ViT)           │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Attention Scores: scores = Q·Kᵀ >>> 3                      │
│  (computed per head, parallelized across lanes)             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Softmax: piece-wise linear approximation                   │
│  (exp LUT + reciprocal LUT for division)                    │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Weighted Sum: output = Σ(attention × V)                    │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Output Projection: Y = concat(heads) · Wo                  │
│  (ternary projection back to model dimension)               │
└─────────────────────────────────────────────────────────────┘
```

## FSM State Machine (LLM Attention)

```verilog
localparam STATE_IDLE       = 4'd0;   // Wait for input
localparam STATE_PROJ_Q     = 4'd1;   // Query projection
localparam STATE_PROJ_KV    = 4'd2;   // Key/Value projection
localparam STATE_ROPE       = 4'd3;   // Apply RoPE
localparam STATE_CACHE_KV   = 4'd4;   // Store K/V in cache
localparam STATE_ATTENTION  = 4'd5;   // Compute attention scores
localparam STATE_SOFTMAX    = 4'd6;   // Softmax normalization
localparam STATE_WEIGHTED   = 4'd7;   // Weighted sum of V
localparam STATE_PROJ_OUT   = 4'd8;   // Output projection
localparam STATE_OUTPUT     = 4'd9;   // Output valid
```

## Saturation and Fixed-Point Arithmetic

All computations use fixed-point with saturation to prevent overflow:

```verilog
parameter ACT_WIDTH = 8;    // 8-bit activations
parameter ACC_WIDTH = 32;   // 32-bit accumulators
parameter FRAC_BITS = 4;    // 4 fractional bits

function signed [ACT_WIDTH-1:0] saturate;
    input signed [ACC_WIDTH-1:0] val;
    begin
        if (val > MAX_POSITIVE)
            saturate = MAX_POSITIVE;
        else if (val < MIN_NEGATIVE)
            saturate = MIN_NEGATIVE;
        else
            saturate = val[ACT_WIDTH-1:0];
    end
endfunction
```

## Hardware Resource Usage

### LLM Attention (per layer)

| Resource | Estimate |
|----------|----------|
| LUTs | ~15K |
| FFs | ~8K |
| BRAM (KV cache) | ~72 × 36Kb blocks |
| DSPs | 0 (ternary eliminates multipliers) |

### ViT Attention (per layer)

| Resource | Estimate |
|----------|----------|
| LUTs | ~20K |
| FFs | ~12K |
| BRAM (token buffer) | ~48 × 36Kb blocks |
| DSPs | 0 |

## Key Optimizations

1. **Ternary Weights** — No multipliers needed, only add/subtract
2. **KV Caching** — O(1) per-token cost for autoregressive generation
3. **Approximate Softmax** — Piece-wise linear avoids exponential hardware
4. **Bit-Shift Scaling** — Division by √64 becomes `>>> 3`
5. **Fixed-Point** — Predictable precision, no floating-point units
6. **Parallel Heads** — Multiple attention heads computed simultaneously

## Testing

Each module includes a testbench (`ifdef SIMULATION`):

```bash
# Run LLM attention testbench
cd rtl/tb
make test_llm_attention

# Run ViT attention testbench
make test_vit_attention

# Run softmax approximation testbench
make test_softmax_approx

# Run KV cache testbench
make test_kv_cache
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original transformer paper
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE paper
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245) — Grouped-Query Attention
- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453) — Ternary weight quantization

---

*See also: [Architecture Overview](ARCHITECTURE.md) | [Quantization Guide](../../model/QUANTIZATION_GUIDE.md)*
