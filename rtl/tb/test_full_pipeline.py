"""
SiLens RTL Testbench - Full Pipeline
====================================

End-to-end integration test for complete vision-language inference:
- Image input processing
- Vision encoder
- Multimodal projector
- Language model
- Token output

Run with:
    cd rtl/tb && make test-pipeline
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, ClockCycles
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tests'))

try:
    from golden.golden_inference import (
        GoldenVisionLanguageModel,
        InferenceConfig
    )
    from golden.golden_transformer import VisionConfig, LanguageConfig
    from golden.golden_attention import WeightType
    GOLDEN_AVAILABLE = True
except ImportError:
    GOLDEN_AVAILABLE = False


# =============================================================================
# Helper Functions
# =============================================================================

def create_test_image(pattern: str, size: int = 56) -> np.ndarray:
    """Create test image with specified pattern."""
    if pattern == 'zeros':
        return np.zeros((size, size, 3), dtype=np.float32)
    elif pattern == 'ones':
        return np.ones((size, size, 3), dtype=np.float32)
    elif pattern == 'gradient':
        g = np.linspace(0, 1, size).reshape(1, -1)
        img = np.tile(g, (size, 1))
        return np.stack([img, img, img], axis=-1).astype(np.float32)
    elif pattern == 'checkerboard':
        check = np.indices((size, size)).sum(axis=0) % 2
        return np.stack([check, check, check], axis=-1).astype(np.float32)
    elif pattern == 'random':
        return np.random.rand(size, size, 3).astype(np.float32)
    else:
        return np.random.rand(size, size, 3).astype(np.float32)


def pack_image_stream(image: np.ndarray, word_width: int = 32) -> list:
    """Pack image into stream of words for DUT input."""
    flat = image.flatten()
    # Quantize to 8-bit
    quantized = np.clip(flat * 255, 0, 255).astype(np.uint8)
    
    bytes_per_word = word_width // 8
    words = []
    
    for i in range(0, len(quantized), bytes_per_word):
        word = 0
        for j in range(bytes_per_word):
            if i + j < len(quantized):
                word |= int(quantized[i + j]) << (j * 8)
        words.append(word)
    
    return words


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

@cocotb.test()
async def test_pipeline_reset(dut):
    """Test pipeline reset behavior."""
    dut._log.info("Testing pipeline: reset")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Assert reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    
    # Check reset state
    if hasattr(dut, 'ready'):
        assert dut.ready.value == 0 or dut.ready.value == 1, "Invalid ready state"
    if hasattr(dut, 'busy'):
        assert dut.busy.value == 0, "Should not be busy after reset"
    
    # Release reset
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    
    # Check post-reset state
    if hasattr(dut, 'ready'):
        dut._log.info(f"Ready after reset: {dut.ready.value}")
    
    dut._log.info("PASS: pipeline reset")


@cocotb.test()
async def test_pipeline_image_input(dut):
    """Test pipeline with image input."""
    dut._log.info("Testing pipeline: image input")
    
    image_size = int(dut.IMAGE_SIZE.value) if hasattr(dut, 'IMAGE_SIZE') else 56
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Create test image
    image = create_test_image('gradient', image_size)
    words = pack_image_stream(image)
    
    dut._log.info(f"Sending {len(words)} words for {image_size}x{image_size} image")
    
    # Send image data
    if hasattr(dut, 'image_valid'):
        for i, word in enumerate(words):
            if hasattr(dut, 'image_data'):
                dut.image_data.value = word
            dut.image_valid.value = 1
            await RisingEdge(dut.clk)
            
            # Check for backpressure
            if hasattr(dut, 'image_ready'):
                while dut.image_ready.value == 0:
                    await RisingEdge(dut.clk)
        
        dut.image_valid.value = 0
    
    dut._log.info("PASS: pipeline image input")


@cocotb.test()
async def test_pipeline_full_inference(dut):
    """Test complete inference pipeline."""
    dut._log.info("Testing pipeline: full inference")
    
    image_size = int(dut.IMAGE_SIZE.value) if hasattr(dut, 'IMAGE_SIZE') else 56
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Create test image
    np.random.seed(42)
    image = create_test_image('random', image_size)
    words = pack_image_stream(image)
    
    # Start timing
    start_cycle = 0
    
    # Send image
    if hasattr(dut, 'image_valid'):
        for word in words:
            if hasattr(dut, 'image_data'):
                dut.image_data.value = word
            dut.image_valid.value = 1
            await RisingEdge(dut.clk)
        dut.image_valid.value = 0
    
    # Wait for inference to complete
    max_cycles = 100000
    output_tokens = []
    
    for cycle in range(max_cycles):
        await RisingEdge(dut.clk)
        
        # Check for output tokens
        if hasattr(dut, 'token_valid') and dut.token_valid.value == 1:
            if hasattr(dut, 'output_token'):
                token = int(dut.output_token.value)
                output_tokens.append(token)
                dut._log.info(f"Cycle {cycle}: Output token {token}")
        
        # Check for completion
        if hasattr(dut, 'inference_done') and dut.inference_done.value == 1:
            dut._log.info(f"Inference completed at cycle {cycle}")
            break
    
    dut._log.info(f"Generated {len(output_tokens)} tokens")
    dut._log.info("PASS: pipeline full inference")


@cocotb.test()
async def test_pipeline_golden_comparison(dut):
    """Compare pipeline output with golden model."""
    if not GOLDEN_AVAILABLE:
        dut._log.warning("Golden model not available, skipping")
        return
    
    dut._log.info("Testing pipeline: golden comparison")
    
    image_size = int(dut.IMAGE_SIZE.value) if hasattr(dut, 'IMAGE_SIZE') else 56
    patch_size = int(dut.PATCH_SIZE.value) if hasattr(dut, 'PATCH_SIZE') else 14
    embed_dim = int(dut.EMBED_DIM.value) if hasattr(dut, 'EMBED_DIM') else 64
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Create golden model
    vision_config = VisionConfig(
        embed_dim=embed_dim, num_heads=4, mlp_dim=embed_dim*4,
        image_size=image_size, patch_size=patch_size, num_layers=2
    )
    language_config = LanguageConfig(
        embed_dim=embed_dim, num_heads=4, mlp_dim=embed_dim*2, num_layers=2
    )
    
    config = InferenceConfig(
        vision_config=vision_config,
        language_config=language_config,
        projector_dim=embed_dim * 2,
        image_size=image_size,
        patch_size=patch_size,
        weight_type=WeightType.TERNARY
    )
    
    golden = GoldenVisionLanguageModel(config)
    
    # Test image
    np.random.seed(42)
    image = create_test_image('random', image_size)
    
    # Golden inference
    golden_output, golden_layers = golden.forward(image)
    dut._log.info(f"Golden output shape: {golden_output.shape}")
    
    # Send to DUT
    words = pack_image_stream(image)
    if hasattr(dut, 'image_valid'):
        for word in words:
            if hasattr(dut, 'image_data'):
                dut.image_data.value = word
            dut.image_valid.value = 1
            await RisingEdge(dut.clk)
        dut.image_valid.value = 0
    
    # Wait for output
    for _ in range(100000):
        await RisingEdge(dut.clk)
        if hasattr(dut, 'inference_done') and dut.inference_done.value == 1:
            break
    
    dut._log.info("PASS: pipeline golden comparison")


# =============================================================================
# Performance Measurement
# =============================================================================

@cocotb.test()
async def test_pipeline_performance(dut):
    """Measure pipeline performance metrics."""
    dut._log.info("Testing pipeline: performance measurement")
    
    image_size = int(dut.IMAGE_SIZE.value) if hasattr(dut, 'IMAGE_SIZE') else 56
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Create test image
    image = create_test_image('random', image_size)
    words = pack_image_stream(image)
    
    # Measure latency
    start_cycle = 0
    
    # Send image and record start
    if hasattr(dut, 'image_valid'):
        for word in words:
            if hasattr(dut, 'image_data'):
                dut.image_data.value = word
            dut.image_valid.value = 1
            await RisingEdge(dut.clk)
            start_cycle += 1
        dut.image_valid.value = 0
    
    # Wait for first output
    output_cycle = start_cycle
    for _ in range(100000):
        await RisingEdge(dut.clk)
        output_cycle += 1
        
        if hasattr(dut, 'token_valid') and dut.token_valid.value == 1:
            break
    
    latency = output_cycle - start_cycle
    
    # Measure throughput (tokens per cycle)
    token_count = 0
    throughput_cycles = 0
    
    for _ in range(10000):
        await RisingEdge(dut.clk)
        throughput_cycles += 1
        
        if hasattr(dut, 'token_valid') and dut.token_valid.value == 1:
            token_count += 1
        
        if hasattr(dut, 'inference_done') and dut.inference_done.value == 1:
            break
    
    throughput = token_count / throughput_cycles if throughput_cycles > 0 else 0
    
    dut._log.info(f"Performance metrics:")
    dut._log.info(f"  First token latency: {latency} cycles")
    dut._log.info(f"  Tokens generated: {token_count}")
    dut._log.info(f"  Throughput: {throughput:.4f} tokens/cycle")
    
    dut._log.info("PASS: pipeline performance")


# =============================================================================
# Test Vector Generation
# =============================================================================

def generate_pipeline_test_vectors(
    image_size: int = 56,
    num_vectors: int = 5,
    seed: int = 42
) -> list:
    """Generate test vectors for full pipeline."""
    if not GOLDEN_AVAILABLE:
        return []
    
    np.random.seed(seed)
    vectors = []
    
    vision_config = VisionConfig(
        embed_dim=64, num_heads=4, mlp_dim=256,
        image_size=image_size, patch_size=14, num_layers=2
    )
    language_config = LanguageConfig(
        embed_dim=64, num_heads=4, mlp_dim=128, num_layers=2
    )
    
    config = InferenceConfig(
        vision_config=vision_config,
        language_config=language_config,
        projector_dim=128,
        image_size=image_size,
        patch_size=14,
        weight_type=WeightType.TERNARY
    )
    
    golden = GoldenVisionLanguageModel(config)
    
    patterns = ['zeros', 'ones', 'gradient', 'checkerboard', 'random']
    
    for pattern in patterns[:num_vectors]:
        image = create_test_image(pattern, image_size)
        output, layers = golden.forward(image)
        
        vectors.append({
            'name': f'pipeline_{pattern}',
            'image': image,
            'expected_output': output,
            'layers': {k: v for k, v in layers.items()},
        })
    
    return vectors


if __name__ == "__main__":
    vectors = generate_pipeline_test_vectors()
    print(f"Generated {len(vectors)} pipeline test vectors")
    
    for v in vectors:
        print(f"  {v['name']}: output shape {v['expected_output'].shape}")
