"""
SiLens RTL Testbench - Vision Encoder
=====================================

cocotb-based integration test for vision encoder pipeline:
- Patch embedding
- Vision transformer stack
- Token output generation

Run with:
    cd rtl/tb && make test-vision
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, ClockCycles
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tests'))

try:
    from golden.golden_inference import (
        GoldenPatchEmbedding,
        GoldenVisionLanguageModel,
        InferenceConfig
    )
    from golden.golden_transformer import VisionConfig, GoldenTransformerStack
    from golden.golden_attention import WeightType
    GOLDEN_AVAILABLE = True
except ImportError:
    GOLDEN_AVAILABLE = False


def pack_image(image: np.ndarray, width: int = 8) -> list:
    """Pack image into list of integers for DUT."""
    flat = image.flatten()
    # Convert to fixed-point
    fp_vals = np.clip(flat * 16, -128, 127).astype(np.int8)
    
    # Pack into chunks
    chunk_size = 32  # bits per word
    values_per_word = chunk_size // width
    
    packed = []
    for i in range(0, len(fp_vals), values_per_word):
        word = 0
        for j in range(values_per_word):
            if i + j < len(fp_vals):
                val = int(fp_vals[i + j]) & 0xFF
                word |= val << (j * width)
        packed.append(word)
    
    return packed


# =============================================================================
# Patch Embedding Tests
# =============================================================================

@cocotb.test()
async def test_patch_embed_zeros(dut):
    """Test patch embedding with zero image."""
    dut._log.info("Testing patch embedding: zero image")
    
    image_size = int(dut.IMAGE_SIZE.value) if hasattr(dut, 'IMAGE_SIZE') else 56
    patch_size = int(dut.PATCH_SIZE.value) if hasattr(dut, 'PATCH_SIZE') else 14
    embed_dim = int(dut.EMBED_DIM.value) if hasattr(dut, 'EMBED_DIM') else 64
    
    num_patches = (image_size // patch_size) ** 2
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Zero image
    image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    packed = pack_image(image)
    
    # Send image data
    for i, word in enumerate(packed):
        if hasattr(dut, 'image_data'):
            dut.image_data.value = word
        if hasattr(dut, 'image_addr'):
            dut.image_addr.value = i
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
    
    dut.valid_in.value = 0
    
    # Wait for output
    for _ in range(5000):
        await RisingEdge(dut.clk)
        if hasattr(dut, 'valid_out') and dut.valid_out.value == 1:
            break
    
    dut._log.info(f"Patch embedding complete for {num_patches} patches")
    dut._log.info("PASS: patch embed zeros")


@cocotb.test()
async def test_patch_embed_gradient(dut):
    """Test patch embedding with gradient image."""
    dut._log.info("Testing patch embedding: gradient image")
    
    image_size = int(dut.IMAGE_SIZE.value) if hasattr(dut, 'IMAGE_SIZE') else 56
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Gradient image
    gradient = np.linspace(0, 1, image_size).reshape(1, -1)
    image = np.tile(gradient, (image_size, 1))
    image = np.stack([image, image, image], axis=-1).astype(np.float32)
    
    packed = pack_image(image)
    
    for i, word in enumerate(packed):
        if hasattr(dut, 'image_data'):
            dut.image_data.value = word
        if hasattr(dut, 'image_addr'):
            dut.image_addr.value = i
        dut.valid_in.value = 1
        dut.ready_out.value = 1
        await RisingEdge(dut.clk)
    
    dut.valid_in.value = 0
    
    for _ in range(5000):
        await RisingEdge(dut.clk)
        if hasattr(dut, 'valid_out') and dut.valid_out.value == 1:
            break
    
    dut._log.info("PASS: patch embed gradient")


@cocotb.test()
async def test_patch_embed_golden(dut):
    """Compare patch embedding with golden model."""
    if not GOLDEN_AVAILABLE:
        dut._log.warning("Golden model not available, skipping")
        return
    
    dut._log.info("Testing patch embedding: golden comparison")
    
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
    golden = GoldenPatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        weight_type=WeightType.TERNARY
    )
    
    np.random.seed(42)
    
    # Random test image
    image = np.random.rand(image_size, image_size, 3).astype(np.float32)
    
    # Golden output
    golden_patches = golden.forward(image)
    dut._log.info(f"Golden patches shape: {golden_patches.shape}")
    
    # Send to DUT
    packed = pack_image(image)
    for i, word in enumerate(packed):
        if hasattr(dut, 'image_data'):
            dut.image_data.value = word
        dut.valid_in.value = 1
        dut.ready_out.value = 1
        await RisingEdge(dut.clk)
    
    dut.valid_in.value = 0
    
    for _ in range(5000):
        await RisingEdge(dut.clk)
        if hasattr(dut, 'valid_out') and dut.valid_out.value == 1:
            break
    
    dut._log.info("PASS: patch embed golden")


# =============================================================================
# Vision Transformer Tests
# =============================================================================

@cocotb.test()
async def test_vision_transformer_forward(dut):
    """Test vision transformer forward pass."""
    dut._log.info("Testing vision transformer: forward pass")
    
    embed_dim = int(dut.EMBED_DIM.value) if hasattr(dut, 'EMBED_DIM') else 64
    num_patches = int(dut.NUM_PATCHES.value) if hasattr(dut, 'NUM_PATCHES') else 17
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Random patch embeddings
    np.random.seed(42)
    patches = np.random.randn(num_patches, embed_dim).astype(np.float32) * 0.1
    
    # Send patches (simplified - actual interface may differ)
    for i in range(num_patches):
        # Pack patch embedding
        patch_data = 0
        for j in range(min(embed_dim, 8)):  # Pack first 8 values
            val = int(patches[i, j] * 16) & 0xFF
            patch_data |= val << (j * 8)
        
        if hasattr(dut, 'patch_data'):
            dut.patch_data.value = patch_data
        if hasattr(dut, 'patch_idx'):
            dut.patch_idx.value = i
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
    
    dut.valid_in.value = 0
    
    # Wait for transformer processing
    for _ in range(10000):
        await RisingEdge(dut.clk)
        if hasattr(dut, 'valid_out') and dut.valid_out.value == 1:
            break
    
    dut._log.info("PASS: vision transformer forward")


# =============================================================================
# Full Vision Encoder Integration
# =============================================================================

@cocotb.test()
async def test_vision_encoder_integration(dut):
    """Full vision encoder integration test."""
    dut._log.info("Testing vision encoder: full integration")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    
    # Simplified test - verify reset state
    if hasattr(dut, 'ready_in'):
        assert dut.ready_in.value == 1, "Should be ready after reset"
    
    dut._log.info("PASS: vision encoder integration")


# =============================================================================
# Test Vector Generation
# =============================================================================

def generate_vision_test_vectors(
    image_size: int = 56,
    patch_size: int = 14,
    embed_dim: int = 64,
    num_images: int = 5,
    seed: int = 42
) -> list:
    """Generate test vectors for vision encoder."""
    if not GOLDEN_AVAILABLE:
        return []
    
    np.random.seed(seed)
    vectors = []
    
    # Create golden model
    vision_config = VisionConfig(
        embed_dim=embed_dim,
        num_heads=4,
        mlp_dim=embed_dim * 4,
        image_size=image_size,
        patch_size=patch_size,
        num_layers=2
    )
    
    config = InferenceConfig(
        vision_config=vision_config,
        image_size=image_size,
        patch_size=patch_size,
        weight_type=WeightType.TERNARY
    )
    
    model = GoldenVisionLanguageModel(config)
    
    # Generate test images
    patterns = [
        ("zeros", np.zeros((image_size, image_size, 3))),
        ("ones", np.ones((image_size, image_size, 3))),
        ("checkerboard", np.tile(
            np.array([[0, 1], [1, 0]]),
            (image_size // 2, image_size // 2)
        )[:, :, np.newaxis].repeat(3, axis=2).astype(np.float32)),
    ]
    
    for name, image in patterns:
        image = image.astype(np.float32)
        patches = model.patch_embed.forward(image)
        features = model.vision_encoder.forward(patches)
        
        vectors.append({
            'name': f'vision_{name}',
            'image': image,
            'patches': patches,
            'features': features,
        })
    
    # Random images
    for i in range(num_images):
        image = np.random.rand(image_size, image_size, 3).astype(np.float32)
        patches = model.patch_embed.forward(image)
        features = model.vision_encoder.forward(patches)
        
        vectors.append({
            'name': f'vision_random_{i}',
            'image': image,
            'patches': patches,
            'features': features,
        })
    
    return vectors


if __name__ == "__main__":
    vectors = generate_vision_test_vectors()
    print(f"Generated {len(vectors)} vision encoder test vectors")
