# SiLens Test Suite

Comprehensive test infrastructure for the SiLens vision-language AI accelerator.

## Overview

The test suite consists of three main components:

1. **Golden Models** (`tests/golden/`) - Pure Python/NumPy reference implementations
2. **RTL Testbenches** (`rtl/tb/`) - cocotb-based hardware simulation tests
3. **Integration Tests** (`tests/integration/`) - End-to-end verification tests

## Directory Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── golden/                  # Golden model reference implementations
│   ├── __init__.py
│   ├── golden_attention.py  # Multi-head attention golden model
│   ├── golden_transformer.py # Transformer block golden model
│   └── golden_inference.py  # Full inference pipeline golden model
├── integration/             # Integration tests
│   ├── __init__.py
│   ├── test_model_accuracy.py      # Model accuracy verification
│   ├── test_sdk_device.py          # SDK device abstraction tests
│   └── test_conversion_pipeline.py # Weight conversion pipeline tests
├── fixtures/                # Test fixtures and data
│   ├── __init__.py
│   ├── test_vectors.py      # Test vector generation
│   └── test_images/         # Sample test images
├── test_model_loading.py    # Model loading tests
├── test_weight_extraction.py # Weight extraction tests
├── test_weights_to_verilog.py # Verilog generation tests
└── README.md                # This file

rtl/tb/                      # RTL testbenches (cocotb)
├── Makefile                 # Test runner
├── test_common.py           # Common module tests (popcount)
├── test_attention.py        # Attention module tests
├── test_transformer_block.py # Transformer block tests
├── test_vision_encoder.py   # Vision encoder tests
├── test_language_model.py   # Language model tests
└── test_full_pipeline.py    # End-to-end pipeline tests
```

## Running Tests

### Quick Start

```bash
# Run all Python tests
make test

# Run specific test file
pytest tests/test_model_loading.py -v

# Run RTL simulation
cd rtl/tb && make sim
```

### Python Tests (pytest)

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=model --cov=tests

# Skip tests requiring model download
pytest tests/ -v --skip-model-tests

# Update golden reference files
pytest tests/ -v --update-golden

# Run only fast tests
pytest tests/ -v -m "not slow"

# Run specific test class
pytest tests/integration/test_model_accuracy.py::TestQuantizationAccuracy -v
```


### RTL Tests (cocotb)

```bash
cd rtl/tb

# Run all RTL tests
make sim

# Run specific module tests
make test-popcount      # Popcount module
make test-ternary-mac   # Ternary MAC
make test-binary-dot    # Binary dot product
make test-softmax       # Softmax approximation
make test-gelu          # GELU activation
make test-layernorm     # Layer normalization

# Run component groups
make test-attention     # All attention components
make test-transformer   # All transformer components

# View waveforms (requires GTKWave)
make waves

# Clean build artifacts
make clean
```

### Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Model accuracy tests (requires model download)
pytest tests/integration/test_model_accuracy.py -v

# SDK device tests (no hardware required)
pytest tests/integration/test_sdk_device.py -v

# Conversion pipeline tests
pytest tests/integration/test_conversion_pipeline.py -v
```

## Golden Models

The golden models provide bit-accurate reference implementations for RTL verification.

### Usage

```python
from tests.golden import (
    GoldenMultiHeadAttention,
    GoldenTransformerBlock,
    GoldenVisionLanguageModel,
    VisionConfig,
    LanguageConfig
)

# Create attention module
attn = GoldenMultiHeadAttention(
    embed_dim=768,
    num_heads=12,
    weight_type='ternary',
    precision='fixed'
)

# Generate test vectors
vectors = attn.generate_test_vectors(num_vectors=100)

# Run forward pass
output, attention_weights = attn.forward(input_tensor, return_attention=True)
```

### Supported Weight Types

- **Binary**: Weights in {-1, +1}, computed using XNOR + popcount
- **Ternary**: Weights in {-1, 0, +1}, computed using MAC with skipping
- **Float**: Full precision (for reference comparison)

### Precision Modes

- **Float**: Standard floating-point computation
- **Fixed**: Fixed-point arithmetic matching RTL implementation

## Test Fixtures

### Generating Test Vectors

```python
from tests.fixtures.test_vectors import (
    generate_reproducible_vectors,
    generate_edge_cases,
    KnownAnswerTest
)

# Generate vectors for a specific module
vectors = generate_reproducible_vectors(
    module='attention',
    num_vectors=100,
    seed=42
)

# Generate edge cases
edge_cases = generate_edge_cases('softmax', seq_len=8)

# Save/load vectors
from tests.fixtures.test_vectors import save_test_vectors, load_test_vectors
save_test_vectors(vectors, 'attention_vectors.json')
loaded = load_test_vectors('attention_vectors.json')
```

### Available Modules

- `popcount`: Population count
- `ternary_mac`: Ternary multiply-accumulate
- `binary_dot`: Binary dot product
- `softmax`: Softmax approximation
- `gelu`: GELU activation
- `layer_norm`: Layer normalization
- `attention`: Multi-head attention
- `transformer`: Transformer block

## Test Markers

Use pytest markers to categorize and filter tests:

```python
@pytest.mark.requires_model   # Requires downloaded model
@pytest.mark.slow             # Long-running test
@pytest.mark.golden           # Golden model comparison test
```

Run with markers:

```bash
pytest tests/ -v -m "not slow"              # Skip slow tests
pytest tests/ -v -m "requires_model"        # Only model tests
pytest tests/ -v -m "golden"                # Only golden tests
```


## Pytest Fixtures

The `conftest.py` provides several useful fixtures:

### Model Fixtures

- `model_path` - Path to the model directory
- `model_available` - Boolean indicating if model is downloaded
- `model_config` - Model configuration dictionary
- `hf_model` - Loaded HuggingFace model (session-scoped)
- `hf_processor` - Loaded HuggingFace processor

### Weight Fixtures

- `weight_extractor` - Utility for extracting model weights

### Test Data Fixtures

- `sample_image` - Sample test image (384x384 RGB)
- `sample_prompt` - Sample text prompt
- `test_data_dir` - Path to test data directory

### Golden Comparison

- `golden_dir` - Path to golden outputs directory
- `golden_comparator` - Utility for comparing against golden references
- `update_golden` - Flag to update golden files (via `--update-golden`)

## Writing New Tests

### Python Tests

1. Create a new test file `test_*.py`
2. Import fixtures from `conftest.py` as needed
3. Use appropriate markers for classification
4. Add golden references with `--update-golden` if needed

```python
import pytest
import numpy as np

class TestMyFeature:
    @pytest.mark.requires_model
    def test_with_model(self, hf_model):
        """Test that requires the model."""
        assert hf_model is not None
    
    def test_standalone(self):
        """Test that doesn't need model."""
        result = my_function()
        assert result is not None
```

### RTL Tests (cocotb)

1. Create test file in `rtl/tb/`
2. Import cocotb and required triggers
3. Use `@cocotb.test()` decorator
4. Add target to Makefile

```python
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

@cocotb.test()
async def test_my_module(dut):
    """Test my RTL module."""
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    
    # Test logic here
    assert dut.output.value == expected
```

## Continuous Integration

The test suite is designed for CI environments:

```bash
# CI-friendly test command (no GPU, no model)
pytest tests/ -v --skip-model-tests

# Full test with model (requires model download)
make model
pytest tests/ -v

# RTL simulation (requires Icarus Verilog)
cd rtl/tb && make sim
```

## Coverage

Generate coverage reports:

```bash
# Python coverage
pytest tests/ --cov=model --cov=tests --cov-report=html

# Open coverage report
open htmlcov/index.html
```

## Troubleshooting

### Model not found
```bash
# Download the model first
make model
```

### cocotb errors
```bash
# Install cocotb
pip install cocotb

# Install Icarus Verilog (macOS)
brew install icarus-verilog

# Install Icarus Verilog (Ubuntu)
sudo apt-get install iverilog
```

### Import errors
```bash
# Add project to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/SiLens
```

## Contributing

When adding new tests:

1. Follow existing naming conventions
2. Add appropriate markers
3. Include docstrings explaining test purpose
4. Use fixtures from `conftest.py` when possible
5. Update this README if adding new test categories
