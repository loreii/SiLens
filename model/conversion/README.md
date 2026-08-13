# SiLens Model Conversion Tools

Complete pipeline for converting SmolVLM-256M to ternary weights for hardware implementation.

## Overview

This directory contains all tools needed to convert the SmolVLM-256M vision-language model from PyTorch format to hardware-ready ternary weights for the SiLens accelerator.

### Conversion Pipeline

```
┌─────────────────┐
│ SmolVLM-256M    │  FP16/FP32 weights from HuggingFace
│ (HuggingFace)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ analyze_model   │  Architecture analysis, statistics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ extract_weights │  Weight extraction, distribution analysis
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ quantize_ternary│  Ternary quantization (-1, 0, +1)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ validate_quant  │  Quality validation, error metrics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ export_config   │  Verilog params, JSON config, C headers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ weights_to_vlog │  Generate Verilog weight modules
└─────────────────┘
```

## Quick Start

### Option 1: Full Pipeline (Recommended)

Run the complete conversion pipeline with a single command:

```bash
python run_pipeline.py --output ./output
```

This will:
1. Download SmolVLM-256M from HuggingFace
2. Extract and analyze all weights
3. Quantize to ternary with optimal alpha
4. Validate quantization quality
5. Generate statistics reports
6. Export Verilog-ready formats


### Option 2: Step-by-Step

Run each step individually for more control:

```bash
# 1. Analyze model architecture
python analyze_model.py --model HuggingFaceTB/SmolVLM-256M-Instruct

# 2. Extract weights with statistics
python extract_weights.py --model HuggingFaceTB/SmolVLM-256M-Instruct --export --output ./weights

# 3. Quantize to ternary
python quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct --alpha 0.7 --export --output ./quantized

# 4. Validate quantization
python validate_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct --quantized ./quantized

# 5. Export configuration
python export_config.py --model HuggingFaceTB/SmolVLM-256M-Instruct --output ./config

# 6. Generate Verilog
python weights_to_verilog.py --weights ./quantized/weights --output ./rtl
```

## Scripts Reference

### `run_pipeline.py` - End-to-End Pipeline

Orchestrates the complete conversion process.

```bash
python run_pipeline.py [options]

Options:
  --model MODEL       Model path or HuggingFace ID (default: HuggingFaceTB/SmolVLM-256M-Instruct)
  --output DIR        Output directory (default: ./model/pipeline_output)
  --alpha FLOAT       Threshold factor for quantization (default: 0.7)
  --mode MODE         Quantization mode: per_tensor, per_channel, per_group
  --skip-download     Skip download, use local model
  --skip-validation   Skip validation step
  --tolerance FLOAT   Validation tolerance (default: 0.1)
  --device DEVICE     Device to use (default: cpu)
```

**Output Structure:**
```
output/
├── extracted/          # Raw extracted weights
├── quantized/          # Ternary quantized weights
│   └── weights/        # Individual weight files
├── validation/         # Validation reports and plots
├── reports/            # Statistics and pipeline results
└── verilog_ready/      # Verilog-ready exports
    ├── vision_encoder/
    ├── projector/
    └── language_model/
```


### `analyze_model.py` - Architecture Analysis

Analyzes model architecture, weight distributions, and hardware feasibility.

```bash
python analyze_model.py [options]

Options:
  --model MODEL    Model path or HuggingFace ID
  --export FILE    Export layer info to JSON
  --verbose        Print detailed per-layer statistics
```

**Output:**
- Parameter counts by component (vision encoder, projector, language model)
- Weight distribution statistics
- Quantization-friendliness analysis
- Hardware resource estimates

### `extract_weights.py` - Weight Extraction

Extracts all weight tensors with comprehensive statistics.

```bash
python extract_weights.py [options]

Options:
  --model MODEL      Model path or HuggingFace ID
  --output DIR       Output directory for weights
  --format FORMAT    Export format: numpy, safetensors, both
  --detailed         Print per-layer statistics
  --visualize        Generate distribution plots
  --recommendations  Print quantization recommendations
```

**Features:**
- Component classification (vision/projector/language model)
- Statistical analysis (mean, std, sparsity, kurtosis)
- Distribution visualization
- Quantization difficulty assessment

### `quantize_ternary.py` - Ternary Quantization

Quantizes FP32 weights to ternary values (-1, 0, +1).

```bash
python quantize_ternary.py [options]

Options:
  --model MODEL     Model path or HuggingFace ID
  --alpha FLOAT     Threshold factor (default: 0.7)
  --mode MODE       per_tensor, per_channel, or per_group
  --output DIR      Output directory
  --export          Export quantized weights
  --search-alpha    Search for optimal alpha value
```

**Quantization Formula:**
```
q(w) = +1  if w > α × mean(|w|)
     = -1  if w < -α × mean(|w|)
     =  0  otherwise
```

**Hardware Encoding:**
| Value | Binary | Hardware |
|-------|--------|----------|
| +1    | 0b01   | Connect to VDD |
| -1    | 0b10   | Connect to GND |
|  0    | 0b00   | No connection |


### `validate_quantization.py` - Quantization Validation

Validates quantization quality by comparing original and quantized outputs.

```bash
python validate_quantization.py [options]

Options:
  --model MODEL          Model path or HuggingFace ID
  --quantized DIR        Path to pre-quantized weights
  --alpha FLOAT          Threshold factor (default: 0.7)
  --tolerance FLOAT      Error tolerance (default: 0.1)
  --detailed             Print per-layer results
  --visualize            Generate error plots
  --output FILE          Export report to JSON
  --inference-samples N  Number of inference samples (default: 10)
```

**Metrics Computed:**
- Mean/Max Absolute Error (MAE)
- Mean/Max Relative Error
- Root Mean Square Error (RMSE)
- Cosine Similarity
- Pearson Correlation

**Quality Assessment:**
| Quality | Cosine Similarity | Expected Accuracy Loss |
|---------|-------------------|------------------------|
| Excellent | > 0.95 | < 3% |
| Good | 0.90 - 0.95 | 3-5% |
| Acceptable | 0.80 - 0.90 | 5-10% |
| Poor | < 0.80 | > 10% |

### `export_config.py` - Configuration Export

Exports model configuration for RTL, SDK, and firmware.

```bash
python export_config.py [options]

Options:
  --model MODEL    Model path or HuggingFace ID
  --output DIR     Output directory
  --format FORMAT  all, json, verilog, c, or layers
```

**Generated Files:**
- `model_config.json` - Complete config for SDK/driver
- `model_params.vh` - Verilog parameters for RTL
- `silens_model_config.h` - C header for firmware
- `layer_info.json` - Detailed layer information


### `weights_to_verilog.py` - Verilog Generation

Converts quantized weights to synthesizable Verilog modules.

```bash
python weights_to_verilog.py [options]

Options:
  -w, --weights DIR    Path to quantized weights
  -o, --output DIR     Output directory for Verilog
  -f, --filter TEXT    Filter layers by name
  --act-width N        Activation bit width (default: 8)
  --acc-width N        Accumulator bit width (default: 32)
  --no-testbench       Skip testbench generation
```

**Generated Verilog Structure:**
```verilog
module layer_name_weights #(
    parameter ACT_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input  wire [ACT_WIDTH-1:0] in [IN_DIM-1:0],
    output wire [ACC_WIDTH-1:0] out [OUT_DIM-1:0]
);
    // +1 weights: assign out[i] = $signed(in[j]);
    // -1 weights: assign out[i] = -$signed(in[j]);
    //  0 weights: No connection
endmodule
```

## SmolVLM-256M Architecture

### Model Components

| Component | Parameters | Description |
|-----------|------------|-------------|
| Vision Encoder (SigLIP) | ~100M | Processes 384×384 images into 729 patch embeddings |
| Projector | ~1.3M | Maps vision embeddings to language model space |
| Language Model (SmolLM) | ~155M | 30-layer transformer for text generation |

### Key Dimensions

```
Vision Encoder:
  Image: 384 × 384 × 3
  Patches: 27 × 27 = 729
  Hidden: 1152
  Layers: 27

Projector:
  Input: 1152
  Output: 576

Language Model:
  Vocab: 49,152
  Hidden: 576
  Layers: 30
  Heads: 9 (3 KV)
```


## Quantization Details

### Ternary Quantization Theory

Ternary quantization maps continuous weights to {-1, 0, +1}:

```
Given weight w and threshold T = α × mean(|W|):

q(w) = sign(w)    if |w| > T
     = 0          otherwise
```

The scale factor for dequantization is the mean of absolute values of non-zero original weights:

```
scale = mean(|W[|W| > T]|)
```

### Choosing Alpha (α)

| Alpha | Sparsity | Accuracy | Use Case |
|-------|----------|----------|----------|
| 0.5 | Low (~20%) | Higher | Maximum accuracy |
| 0.7 | Medium (~35%) | Balanced | **Recommended** |
| 0.9 | High (~50%) | Lower | Maximum compression |

### Per-Channel vs Per-Tensor

- **Per-Tensor**: Single threshold for entire weight matrix
  - Simpler hardware
  - Slightly lower accuracy
  
- **Per-Channel**: Threshold per output channel
  - Better handles varying scales
  - Slightly more complex
  - **Recommended for critical layers**

## Expected Quality

### Accuracy Benchmarks

| Benchmark | FP16 | Ternary (α=0.7) | Degradation |
|-----------|------|-----------------|-------------|
| VQAv2 | 71% | 65-68% | 3-6% |
| TextVQA | 55% | 48-52% | 3-7% |
| COCO Caption | 0.85 BLEU | 0.78-0.82 | 3-8% |

### Memory Savings

| Format | Bits/Weight | Memory (256M params) |
|--------|-------------|---------------------|
| FP32 | 32 | 1024 MB |
| FP16 | 16 | 512 MB |
| INT8 | 8 | 256 MB |
| **Ternary** | **2** | **64 MB** |

Compression: **16× vs FP32**, **8× vs FP16**


## Hardware Considerations

### Hardwired Weight Implementation

For the SiLens accelerator, ternary weights are implemented as hardwired connections:

```
+1: Wire input directly to accumulator (add)
-1: Wire through inverter to accumulator (subtract)
 0: No connection (contributes nothing)
```

This eliminates memory access for weights entirely!

### Resource Estimates

For 256M parameters:
- **Non-zero weights**: ~160M (assuming 35% sparsity)
- **Transistors/weight**: ~6 (for routing/connection)
- **Total transistors**: ~1B
- **Estimated area**: 200-400 mm² (SKY130)

### KV Cache Requirements

For language model inference:
- **Per token**: 30 layers × 2 (K+V) × 192 channels = 11.5 KB
- **2K context**: ~23 MB
- **8K context**: ~92 MB

## Validation Checklist

Before deploying to hardware:

- [ ] Model downloads successfully
- [ ] Weight statistics look reasonable (symmetric, no extreme outliers)
- [ ] Quantization completes without errors
- [ ] Validation cosine similarity > 0.90
- [ ] Per-layer errors within tolerance
- [ ] Statistics report shows expected sparsity (30-40%)
- [ ] Verilog generates without syntax errors
- [ ] RTL simulation matches PyTorch output

## Troubleshooting

### Common Issues

**"transformers not installed"**
```bash
pip install transformers torch
```

**"Model not found"**
- Check internet connection
- Verify model ID: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Try downloading manually first

**Low cosine similarity (<0.85)**
- Try lower alpha (0.6 instead of 0.7)
- Use per-channel quantization
- Check for outlier layers in detailed report

**High sparsity (>50%)**
- Alpha is too high
- Reduce alpha to 0.6-0.7

**Verilog synthesis errors**
- Check for very large layers (may need splitting)
- Verify bit widths match your toolchain

## Dependencies

Required Python packages:
```
torch>=2.0.0
transformers>=4.36.0
numpy>=1.21.0
```

Optional (for visualization):
```
matplotlib>=3.5.0
seaborn>=0.11.0
```

Install all:
```bash
pip install torch transformers numpy matplotlib seaborn
```

## License

Apache 2.0 - See LICENSE file in project root.

## Contributing

See CONTRIBUTING.md in project root for guidelines.
