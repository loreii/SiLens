# SiLens Python SDK

Python interface for the SiLens Vision-Language AI Accelerator.

## Overview

The SiLens SDK provides a high-level Python API for interfacing with SiLens hardware accelerators. It supports:

- **PCIe Mode**: Direct communication with SiLens PCIe cards
- **USB Mode**: Connection to FPGA prototypes via USB
- **Simulation Mode**: Development and testing without hardware

## Features

- 🚀 **High-performance inference** on SiLens hardware
- 🔄 **Streaming output** for real-time token generation
- ⚡ **Async support** with asyncio for non-blocking operations
- 🖼️ **Image preprocessing** with patch extraction for ViT
- 📝 **Tokenization** wrapper for SmolLM2
- 📊 **Benchmarking** utilities for performance analysis
- 💻 **CLI tools** for quick testing and diagnostics

## Installation

### From PyPI (when published)

```bash
pip install silens
```

### From Source

```bash
git clone https://github.com/silens/silens-sdk
cd silens-sdk/sdk
pip install -e ".[full]"
```

### Dependencies

- **Required**: `numpy`, `pillow`
- **Optional**: `transformers`, `torch` (for full tokenizer support)
- **USB devices**: `pyusb`

## Quick Start

### Basic Usage

```python
from silens import SiLensDevice, InferenceEngine

# Discover and connect to device (uses simulation if no hardware)
device = SiLensDevice.discover()[0]

with device:
    # Create inference engine
    engine = InferenceEngine(device)
    
    # Describe an image
    result = engine.describe_image("photo.jpg")
    print(result.text)
    
    # With custom prompt
    result = engine.run("photo.jpg", "What color is the car?")
    print(result.text)
    
    # Print statistics
    print(result.summary())
```

### Streaming Output

```python
from silens import SiLensDevice, InferenceEngine

device = SiLensDevice.discover()[0]

with device:
    engine = InferenceEngine(device)
    
    # Stream tokens as they are generated
    for token in engine.stream("photo.jpg", "Describe this image"):
        print(token, end="", flush=True)
```

### Async Support

```python
import asyncio
from silens import SiLensDevice, InferenceEngine

async def main():
    device = SiLensDevice.discover()[0]
    
    with device:
        engine = InferenceEngine(device)
        
        # Async inference
        result = await engine.run_async("photo.jpg", "Describe this image")
        print(result.text)
        
        # Async streaming
        async for token in engine.stream_async("photo.jpg", "What objects?"):
            print(token, end="", flush=True)

asyncio.run(main())
```

### Simulation Mode

```python
from silens import SiLensDevice, SimulatedDevice, InferenceEngine

# Force simulation mode
devices = SiLensDevice.discover(mode="simulation")
device = devices[0]

# Or create simulated device directly
device = SimulatedDevice(
    latency_ms=10.0,      # Simulated vision processing time
    tokens_per_sec=50.0   # Simulated token generation rate
)

with device:
    engine = InferenceEngine(device)
    result = engine.describe_image("photo.jpg")
```

## Command-Line Interface

The SDK includes a powerful CLI for common tasks:

```bash
# Show device information
silens info
silens info --verbose

# Run inference on an image
silens infer photo.jpg
silens infer photo.jpg --prompt "What objects are visible?"
silens infer photo.jpg --stream --max-tokens 128

# Run performance benchmark
silens benchmark photo.jpg --iterations 100
silens benchmark --output results.json

# Convert models (coming soon)
silens convert HuggingFaceTB/SmolLM2-135M --quantize
```

## API Reference

### SiLensDevice

Abstract base class for device interfaces.

```python
class SiLensDevice:
    @classmethod
    def discover(mode="auto") -> List[SiLensDevice]
        """Discover available devices.
        
        Args:
            mode: "auto", "pcie", "usb", or "simulation"
        """
    
    def open() -> None
        """Open connection to device."""
    
    def close() -> None
        """Close connection."""
    
    def read_reg(offset: int) -> int
        """Read 32-bit register."""
    
    def write_reg(offset: int, value: int) -> None
        """Write 32-bit register."""
    
    def get_version() -> tuple[int, int, int]
        """Get hardware version (major, minor, patch)."""
    
    def is_ready() -> bool
        """Check if device is ready."""
    
    def reset() -> None
        """Reset the device."""
```


### InferenceEngine

High-level inference interface.

```python
class InferenceEngine:
    def __init__(
        device: SiLensDevice,
        model_config: ModelConfig = None,
        tokenizer_name: str = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    )
    
    def describe_image(image, prompt="Describe this image.") -> InferenceResult
        """Generate description for an image."""
    
    def run(image, prompt, **kwargs) -> InferenceResult
        """Run inference with custom prompt."""
    
    def stream(image, prompt, **kwargs) -> Iterator[str]
        """Stream tokens as generated."""
    
    # Async methods
    async def run_async(image, prompt, **kwargs) -> InferenceResult
        """Run inference asynchronously."""
    
    async def describe_image_async(image, prompt, **kwargs) -> InferenceResult
        """Describe image asynchronously."""
    
    async def stream_async(image, prompt, **kwargs) -> AsyncIterator[str]
        """Stream tokens asynchronously."""
```

### InferenceResult

```python
@dataclass
class InferenceResult:
    text: str                    # Generated text
    tokens: List[int]            # Token IDs
    num_tokens: int              # Number of tokens
    vision_time_ms: float        # Vision encoder time
    generation_time_ms: float    # Token generation time
    total_time_ms: float         # Total time
    tokens_per_second: float     # Generation speed
    
    def summary() -> str
        """Get formatted summary."""
```

### ImageProcessor

```python
from silens import ImageProcessor, ImageConfig

config = ImageConfig(
    size=384,           # Target image size
    patch_size=16,      # Patch size for ViT
    normalize=True,     # Apply normalization
    center_crop=False,  # Center crop to square
)

processor = ImageProcessor(config)

# Process single image
pixels = processor.process("photo.jpg")

# Process and get patches
pixels, patches = processor.process("photo.jpg", return_patches=True)

# Process batch
batch = processor.process_batch(["img1.jpg", "img2.jpg"])
```

### SiLensTokenizer

```python
from silens import SiLensTokenizer, TokenizerConfig

config = TokenizerConfig(
    name="HuggingFaceTB/SmolLM2-135M",
    max_length=8192,
)

tokenizer = SiLensTokenizer(config)

# Encode/decode
tokens = tokenizer.encode("Hello world")
text = tokenizer.decode(tokens)

# Format VLM prompt
prompt = tokenizer.format_vlm_prompt(
    "What is in this image?",
    num_image_tokens=576,
)

# Chat formatting
messages = [
    {"role": "user", "content": "What do you see?"},
]
formatted = tokenizer.format_chat(messages)
```

### Benchmark

```python
from silens import Benchmark, quick_benchmark

# Quick benchmark
result = quick_benchmark(engine, "photo.jpg", iterations=10)
result.print_summary()

# Detailed benchmark
benchmark = Benchmark(engine, name="my-benchmark")
result = benchmark.run_latency(
    image="photo.jpg",
    iterations=100,
    warmup=10,
    max_new_tokens=64,
)
result.save("results.json")

# Throughput benchmark
result = benchmark.run_throughput(images_list, prompt)
```

### ModelConfig

```python
from silens import ModelConfig, load_model_config

# Load default config
config = load_model_config()

# Load from file
config = load_model_config("model_config.json")

# Access parameters
print(f"Model: {config.name}")
print(f"Vision encoder: {config.vision.hidden_dim}d, {config.vision.num_layers} layers")
print(f"Language model: {config.language.hidden_dim}d, {config.language.num_layers} layers")
print(f"Total parameters: {config.total_parameters_str}")
```

## Hardware Register Map

The SDK communicates with hardware via memory-mapped registers:

| Offset | Name | Description |
|--------|------|-------------|
| 0x000 | CTRL | Control register |
| 0x004 | STATUS | Status register |
| 0x008 | IMG_ADDR | Image buffer address |
| 0x00C | IMG_SIZE | Image dimensions |
| 0x010 | OUT_ADDR | Output buffer address |
| 0x014 | OUT_LEN | Output length |
| 0x018 | TOKEN_OUT | Current output token |
| 0x01C | TOKEN_VALID | Token valid flag |
| 0x100 | DMA_CTRL | DMA control |
| 0x1F0 | VERSION | Hardware version |


## Examples

### Run from Command Line

```bash
# Basic inference
python examples/simple_inference.py photo.jpg

# Custom prompt
python examples/simple_inference.py photo.jpg -p "What objects are visible?"

# Streaming output
python examples/simple_inference.py photo.jpg --stream

# Benchmark mode
python examples/simple_inference.py photo.jpg --benchmark -n 100

# Simulation mode (no hardware)
python examples/simple_inference.py photo.jpg --simulation
```

### Batch Processing

```bash
python examples/batch_inference.py images/ --output results.json
```

```python
from silens import SiLensDevice, InferenceEngine
from pathlib import Path

device = SiLensDevice.discover()[0]

with device:
    engine = InferenceEngine(device)
    
    images = Path("images/").glob("*.jpg")
    
    for image_path in images:
        result = engine.describe_image(str(image_path))
        print(f"{image_path.name}: {result.text[:100]}...")
```

### Video Captioning

```bash
python examples/video_captioning.py video.mp4 --fps 1 --output captions.srt
```

### Interactive Chat

```bash
python examples/interactive_chat.py --image photo.jpg
```

### Benchmark Comparison

```bash
python examples/benchmark_comparison.py --iterations 50 --with-gpu --output results.json
```

### Custom Preprocessing

```python
from silens import SiLensDevice, InferenceEngine, ImageProcessor, ImageConfig
import numpy as np

# Custom preprocessing config
config = ImageConfig(
    size=384,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    center_crop=True,
)
processor = ImageProcessor(config)

# Process image
processed = processor.process("photo.jpg")

# Run inference
device = SiLensDevice.discover()[0]
with device:
    engine = InferenceEngine(device)
    result = engine.run(processed, "Describe this image")
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Formatting

```bash
black silens/
isort silens/
flake8 silens/
mypy silens/
```

## Architecture

```
sdk/
├── silens/
│   ├── __init__.py          # Package exports
│   ├── device.py            # Device abstraction (PCIe, USB, Simulation)
│   ├── inference.py         # High-level inference API
│   ├── model.py             # Model configuration
│   ├── utils.py             # Utilities (image, timing, memory)
│   ├── image_processing.py  # Image preprocessing and patches
│   ├── tokenizer.py         # Tokenization wrapper
│   ├── streaming.py         # Streaming inference support
│   ├── benchmark.py         # Performance benchmarking
│   └── cli.py               # Command-line interface
├── examples/
│   ├── simple_inference.py
│   ├── batch_inference.py
│   ├── video_captioning.py
│   ├── interactive_chat.py
│   └── benchmark_comparison.py
├── docs/
│   └── QUICKSTART.md
├── tests/
├── pyproject.toml
└── README.md
```

## License

Apache License 2.0

## Support

- GitHub Issues: https://github.com/silens/silens-sdk/issues
- Documentation: https://docs.silens.ai
- Discord: https://discord.gg/silens
