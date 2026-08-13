# SiLens SDK Quick Start Guide

Get up and running with SiLens in under 5 minutes.

## Installation

### From PyPI (when published)

```bash
pip install silens
```

### From Source

```bash
git clone https://github.com/silens/silens-sdk
cd silens-sdk/sdk
pip install -e .
```

### With All Dependencies

```bash
# Full installation with tokenizers and ML frameworks
pip install -e ".[full]"

# Development installation
pip install -e ".[dev]"
```

## Quick Start

### 1. Basic Inference

```python
from silens import SiLensDevice, InferenceEngine

# Discover and connect to device
# (Automatically uses simulation mode if no hardware is present)
device = SiLensDevice.discover()[0]

with device:
    # Create inference engine
    engine = InferenceEngine(device)
    
    # Describe an image
    result = engine.describe_image("photo.jpg")
    print(result.text)
    
    # With a custom prompt
    result = engine.run("photo.jpg", "What color is the car?")
    print(result.text)
```

### 2. Streaming Output

```python
from silens import SiLensDevice, InferenceEngine

device = SiLensDevice.discover()[0]

with device:
    engine = InferenceEngine(device)
    
    # Stream tokens as generated
    for token in engine.stream("photo.jpg", "Describe this image"):
        print(token, end="", flush=True)
    print()  # Newline at end
```


### 3. Async Support

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
        async for token in engine.stream_async("photo.jpg", "What is in this image?"):
            print(token, end="", flush=True)

asyncio.run(main())
```

### 4. Simulation Mode

For development without hardware:

```python
from silens import SiLensDevice, SimulatedDevice, InferenceEngine

# Option 1: Force simulation mode via discover
devices = SiLensDevice.discover(mode="simulation")
device = devices[0]

# Option 2: Create simulated device directly
device = SimulatedDevice(
    latency_ms=10.0,      # Simulated vision processing time
    tokens_per_sec=50.0   # Simulated token generation rate
)

with device:
    engine = InferenceEngine(device)
    result = engine.describe_image("photo.jpg")
```

## Command-Line Interface

The SDK includes a powerful CLI:

```bash
# Show device info
silens info

# Run inference on an image
silens infer photo.jpg
silens infer photo.jpg --prompt "What objects are visible?"
silens infer photo.jpg --stream

# Run benchmarks
silens benchmark photo.jpg --iterations 100
silens benchmark --output results.json

# Convert models (coming soon)
silens convert HuggingFaceTB/SmolLM2-135M --output converted_model/
```

## Examples

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
    
    for image in Path("images/").glob("*.jpg"):
        result = engine.describe_image(str(image))
        print(f"{image.name}: {result.text[:100]}...")
```


### Video Captioning

```bash
python examples/video_captioning.py video.mp4 --fps 1 --output captions.srt
```

### Interactive Chat

```bash
python examples/interactive_chat.py --image photo.jpg
```

### Benchmarking

```bash
python examples/benchmark_comparison.py --iterations 50 --with-gpu
```

## Image Preprocessing

```python
from silens import ImageProcessor, ImageConfig

# Create processor with default settings
processor = ImageProcessor()

# Or customize
config = ImageConfig(
    size=384,
    patch_size=16,
    center_crop=True,
    normalize=True,
)
processor = ImageProcessor(config)

# Process an image
pixels = processor.process("photo.jpg")

# Get patches for ViT
pixels, patches = processor.process("photo.jpg", return_patches=True)
print(f"Patches shape: {patches.shape}")  # (576, 768)
```

## Tokenization

```python
from silens import SiLensTokenizer, get_tokenizer

# Create tokenizer
tokenizer = SiLensTokenizer()

# Encode text
tokens = tokenizer.encode("Describe this image in detail.")
print(f"Tokens: {tokens}")

# Decode tokens
text = tokenizer.decode(tokens)
print(f"Text: {text}")

# Format VLM prompt
prompt = tokenizer.format_vlm_prompt(
    "What objects are in this image?",
    num_image_tokens=576,
)
```

## Benchmarking

```python
from silens import SiLensDevice, InferenceEngine, Benchmark, quick_benchmark

device = SiLensDevice.discover()[0]

with device:
    engine = InferenceEngine(device)
    
    # Quick benchmark
    result = quick_benchmark(engine, "photo.jpg", iterations=10)
    result.print_summary()
    
    # Detailed benchmark
    benchmark = Benchmark(engine)
    result = benchmark.run_latency(
        image="photo.jpg",
        iterations=100,
        warmup=10,
    )
    result.save("benchmark_results.json")
```

## Device Info

```python
from silens import SiLensDevice

# Discover all devices
devices = SiLensDevice.discover()

for device in devices:
    print(f"Device: {device}")
    
    with device:
        version = device.get_version()
        print(f"  Version: {version[0]}.{version[1]}.{version[2]}")
        print(f"  Ready: {device.is_ready()}")
```

## Next Steps

- Read the [full API documentation](../README.md)
- Explore the [example scripts](../examples/)
- Check out the [model configuration](../silens/model.py)
- See the [hardware driver documentation](../../drivers/README.md)

## Troubleshooting

### No Device Found

```python
# Force simulation mode
devices = SiLensDevice.discover(mode="simulation")
```

### Missing Dependencies

```bash
# Install full dependencies
pip install silens[full]

# Or specific extras
pip install silens[usb]  # For USB devices
```

### Slow Performance

- Ensure you're using hardware mode, not simulation
- Check `device.get_status()` for busy/error states
- Reduce `max_new_tokens` for faster responses

## Support

- GitHub Issues: https://github.com/silens/silens-sdk/issues
- Documentation: https://docs.silens.ai
- Discord: https://discord.gg/silens
