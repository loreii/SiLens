# SiLens Ternary Quantization Guide

A comprehensive guide for achieving optimal quantization results when converting SmolVLM-256M to ternary weights for the SiLens accelerator.

## Table of Contents

1. [Introduction](#introduction)
2. [Quantization Basics](#quantization-basics)
3. [Choosing Optimal Settings](#choosing-optimal-settings)
4. [Advanced Techniques](#advanced-techniques)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

---

## Introduction

Ternary quantization converts 32-bit floating-point weights to just 2 bits per weight ({-1, 0, +1}), enabling:

- **16x memory reduction** compared to FP32
- **Hardwired implementation** on SiLens accelerator
- **Power-efficient inference** with simple add/subtract operations

This guide covers the optimal settings and techniques for minimizing accuracy loss during quantization.

---

## Quantization Basics

### The Ternary Quantization Formula

```
q(w) = +1  if w > τ
     = -1  if w < -τ  
     =  0  otherwise

where: τ = α × mean(|W|)
```

**Parameters:**
- **α (alpha)**: Threshold factor, controls sparsity
- **τ (threshold)**: Actual threshold value
- **scale**: Factor for dequantization

### Hardware Encoding

| Value | Binary | Hardware Implementation |
|-------|--------|------------------------|
| +1 | 0b01 | Connect to VDD (add) |
| -1 | 0b10 | Connect to GND (subtract) |
|  0 | 0b00 | No connection (skip) |

### Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| Cosine Similarity | Output alignment | > 0.90 |
| Mean Absolute Error | Weight reconstruction | < 0.05 |
| Sparsity | Fraction of zeros | 25-40% |
| Perplexity Increase | Language model quality | < 20% |

---

## Choosing Optimal Settings

### Alpha (α) Selection

The alpha parameter is the most important choice. It directly controls the sparsity-accuracy tradeoff.

| Alpha | Sparsity | Accuracy | Use Case |
|-------|----------|----------|----------|
| 0.5 | ~20% | Highest | Maximum accuracy, critical applications |
| 0.6 | ~27% | High | Good balance for sensitive layers |
| **0.7** | **~35%** | **Balanced** | **Recommended default** |
| 0.8 | ~42% | Medium | Higher compression, some accuracy loss |
| 0.9 | ~50% | Lower | Maximum compression |

**How to choose:**

```bash
# Option 1: Search optimal alpha
python quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct --search-alpha

# Option 2: Gradient-based optimization (best accuracy)
python quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct --optimize-alpha
```

### Quantization Mode

| Mode | Description | Pros | Cons |
|------|-------------|------|------|
| **per_tensor** | Single threshold per layer | Simple, hardware-friendly | Lower accuracy |
| **per_channel** | Threshold per output channel | Better accuracy | Slightly more complex |
| **per_group** | Threshold per weight group | Best accuracy | Most complex |

**Recommendations:**

- Use **per_tensor** for simpler hardware and faster inference
- Use **per_channel** for better accuracy with minimal complexity
- Use **per_group** only if accuracy is critical

```bash
# Per-channel quantization (recommended for accuracy)
python quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct --mode per_channel
```

---

## Advanced Techniques

### 1. Calibration-Aware Quantization

Calibration uses sample data to find optimal thresholds that minimize actual output error.

```bash
python calibration.py --model HuggingFaceTB/SmolVLM-256M-Instruct --samples 200
```

**When to use:**
- When simple alpha selection isn't achieving target accuracy
- For production deployments requiring consistent quality
- When you have representative calibration data

### 2. Mixed-Precision Quantization

Keep critical layers at higher precision while using ternary for the majority.

```bash
python mixed_precision.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --high-precision-ratio 0.1 --high-precision-level int8
```

**Recommended high-precision layers:**
- Embedding layers
- Language model head (lm_head)
- First/last attention layers

### 3. Sensitivity Analysis

Identify which layers are most affected by quantization.

```bash
python sensitivity_analysis.py --model HuggingFaceTB/SmolVLM-256M-Instruct --samples 50
```

Use sensitivity results to:
- Apply per-layer optimal alpha
- Decide which layers need higher precision
- Focus calibration efforts

### 4. Outlier Handling

Large outlier weights can distort quantization. Pre-clip them for better results.

```bash
# Detect outliers
python analysis/outlier_detector.py --model HuggingFaceTB/SmolVLM-256M-Instruct

# If significant outliers detected, use percentile clipping
python calibration.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --percentile-clip 99.5
```

---

## Troubleshooting

### Problem: Very High Sparsity (>50%)

**Symptoms:** Many layers have >50% zeros, accuracy drops significantly

**Causes:**
- Alpha too high
- Outliers inflating mean(|w|)

**Solutions:**
1. Lower alpha to 0.5-0.6
2. Apply outlier clipping before quantization
3. Use per-channel quantization

### Problem: Low Cosine Similarity (<0.85)

**Symptoms:** Quantized outputs diverge from original

**Causes:**
- Some layers are very sensitive to quantization
- Weight distributions are highly non-normal

**Solutions:**
1. Run sensitivity analysis to identify problematic layers
2. Use mixed-precision for sensitive layers
3. Use gradient-based alpha optimization

### Problem: Specific Layer Failures

**Symptoms:** One or few layers show very poor metrics

**Causes:**
- Layer has unusual weight distribution
- Layer uses outliers for important features

**Solutions:**
1. Check layer with outlier detector
2. Use per-channel quantization for that layer
3. Keep layer at higher precision (int8)

### Problem: Vision vs Language Quality Mismatch

**Symptoms:** Vision tasks ok, language tasks degraded (or vice versa)

**Causes:**
- Vision and language model parts have different sensitivities

**Solutions:**
1. Use separate alpha for each component
2. Run sensitivity analysis per component
3. Consider mixed precision for the weaker component

---

## Best Practices

### Pre-Quantization Checklist

- [ ] Analyze model weights: `python analyze_model.py`
- [ ] Check for outliers: `python analysis/outlier_detector.py`
- [ ] Run sensitivity analysis: `python sensitivity_analysis.py`
- [ ] Visualize distributions: `python analysis/weight_visualizer.py`

### Recommended Workflow

```bash
# Step 1: Analyze model
python analyze_model.py --model HuggingFaceTB/SmolVLM-256M-Instruct

# Step 2: Detect potential issues
python analysis/outlier_detector.py --model HuggingFaceTB/SmolVLM-256M-Instruct
python analysis/sparsity_analyzer.py --model HuggingFaceTB/SmolVLM-256M-Instruct

# Step 3: Run sensitivity analysis
python sensitivity_analysis.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --output ./sensitivity.json

# Step 4: Choose quantization approach based on results
# If sensitivity is uniform: Use simple quantization with alpha=0.7
# If sensitivity varies: Use mixed precision or per-layer alphas

# Step 5: Quantize with chosen settings
python quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --alpha 0.7 --mode per_tensor --export --output ./quantized

# Step 6: Validate results
python validate_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
    --quantized ./quantized --detailed

# Step 7: Run benchmarks
python validation/benchmark_suite.py --model HuggingFaceTB/SmolVLM-256M-Instruct --all
```

### Quality Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Avg Cosine Similarity | 0.80 | 0.90 | 0.95 |
| VQA Accuracy (relative) | 85% | 92% | 97% |
| Perplexity Increase | <30% | <15% | <5% |
| Sparsity | 20-50% | 30-40% | 32-38% |

### Hardware Considerations

**Memory:**
- Ternary weights: 2 bits/weight = 64 MB for 256M params
- Scale factors: Add ~1% overhead
- Activation memory separate (configurable)

**KV Cache:**
- Keep at 8-bit for quality
- ~11.5 KB per token
- 2K context = ~23 MB

**Inference Speed:**
- Ternary enables add/subtract only (no multiply)
- Sparsity allows skipping zero weights
- Target: >10x faster than FP32

---

## Tool Reference

### Quantization Tools

| Tool | Purpose | Key Options |
|------|---------|-------------|
| `quantize_ternary.py` | Main quantization | `--alpha`, `--mode`, `--optimize-alpha` |
| `calibration.py` | Calibration-aware quant | `--samples`, `--percentile-clip` |
| `mixed_precision.py` | Multi-precision quant | `--high-precision-ratio` |
| `sensitivity_analysis.py` | Layer sensitivity | `--samples`, `--output` |

### Validation Tools

| Tool | Purpose | Key Options |
|------|---------|-------------|
| `validate_quantization.py` | Weight comparison | `--detailed`, `--tolerance` |
| `benchmark_suite.py` | VQA/TextVQA benchmarks | `--benchmark`, `--samples` |
| `perplexity_test.py` | LM perplexity | `--dataset`, `--compare` |
| `visual_qa_test.py` | VQA accuracy | `--samples`, `--compare` |
| `compare_outputs.py` | Output comparison | `--detailed`, `--visualize` |

### Analysis Tools

| Tool | Purpose | Key Options |
|------|---------|-------------|
| `weight_visualizer.py` | Distribution plots | `--plot`, `--alpha` |
| `sparsity_analyzer.py` | Sparsity patterns | `--plot`, `--output` |
| `outlier_detector.py` | Outlier detection | `--method`, `--threshold` |

---

## Appendix: Expected Results

### SmolVLM-256M Benchmarks

**With default settings (α=0.7, per_tensor):**

| Metric | Original | Quantized | Change |
|--------|----------|-----------|--------|
| VQA Accuracy | ~71% | ~66-68% | -3-5% |
| TextVQA | ~55% | ~50-52% | -3-5% |
| Perplexity | ~15 | ~17-18 | +13-20% |
| Cosine Similarity | 1.0 | 0.91-0.94 | - |

**With optimization (calibration + mixed precision):**

| Metric | Original | Quantized | Change |
|--------|----------|-----------|--------|
| VQA Accuracy | ~71% | ~68-70% | -1-3% |
| TextVQA | ~55% | ~52-54% | -1-3% |
| Perplexity | ~15 | ~15.5-16.5 | +3-10% |
| Cosine Similarity | 1.0 | 0.94-0.97 | - |

---

## Getting Help

- **GitHub Issues:** Report bugs or ask questions
- **Documentation:** Check tool docstrings and `--help` flags
- **Examples:** See `model/conversion/README.md` for detailed examples

## License

Apache 2.0 - See LICENSE file in project root.
