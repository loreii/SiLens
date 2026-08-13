# SiLens Test Suite

This directory contains the test infrastructure for the SiLens project.

## Test Categories

### Python Tests (pytest)
- **test_model_loading.py** - Tests for SmolVLM-256M model loading and inference
- **test_weight_extraction.py** - Tests for weight extraction and quantization pipeline

### RTL Tests (cocotb)
Located in `rtl/tb/`:
- **test_common.py** - Tests for common RTL modules (popcount, etc.)

## Running Tests

### Quick Start

```bash
# Run all Python tests
make test

# Run RTL simulation
make sim
```

### Python Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model_loading.py -v

# Skip tests requiring model download
pytest tests/ -v --skip-model-tests

# Update golden reference files
pytest tests/ -v --update-golden

# Run only fast tests (skip slow ones)
pytest tests/ -v -m "not slow"
```

### RTL Tests (cocotb)

```bash
cd rtl/tb

# Run all RTL tests
make sim

# Run specific module test
make test-popcount

# View waveforms
make waves

# Clean build artifacts
make clean
```

## Test Fixtures

The `conftest.py` provides several useful fixtures:

### Model Fixtures
- `model_path` - Path to the model directory
- `model_available` - Boolean indicating if model is downloaded
- `hf_model` - Loaded HuggingFace model (session-scoped)
- `hf_processor` - Loaded HuggingFace processor

### Weight Fixtures
- `weight_extractor` - Utility for extracting model weights

### Test Data Fixtures
- `sample_image` - Sample test image (384x384 RGB)
- `sample_prompt` - Sample text prompt

### Golden Comparison
- `golden_comparator` - Utility for comparing against golden references

## Test Markers

- `@pytest.mark.requires_model` - Tests that need the model downloaded
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.golden` - Tests that compare against golden references

## Directory Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_model_loading.py    # Model loading tests
├── test_weight_extraction.py # Weight extraction tests
├── test_data/               # Test input files
├── golden/                  # Golden reference outputs
└── README.md               # This file
```

## Adding New Tests

1. Create a new test file `test_*.py`
2. Import fixtures from `conftest.py` as needed
3. Use appropriate markers for test classification
4. Add golden references with `--update-golden` if needed

## Continuous Integration

The test suite is designed to work in CI environments:

```bash
# CI-friendly test command (no GPU required)
pytest tests/ -v --skip-model-tests

# Full test with model (requires model download)
make model
pytest tests/ -v
```
