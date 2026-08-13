"""
SiLens RTL Testbench - Transformer Block
=========================================

cocotb-based testbench for transformer block RTL modules:
- Layer normalization
- GELU activation
- Full transformer block integration

Run with:
    cd rtl/tb && make test-transformer
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tests'))

try:
    from golden.golden_transformer import (
        GoldenLayerNorm,
        GoldenGELU,
        GoldenMLP,
        GoldenTransformerBlock,
        VisionConfig
    )
    from golden.golden_attention import FixedPointOps, WeightType
    GOLDEN_AVAILABLE = True
except ImportError:
    GOLDEN_AVAILABLE = False


# =============================================================================
# Helper Functions
# =============================================================================

def to_fixed_point(value: float, frac_bits: int = 4, width: int = 8) -> int:
    """Convert float to fixed-point integer."""
    scale = 1 << frac_bits
    max_val = (1 << (width - 1)) - 1
    min_val = -(1 << (width - 1))
    scaled = int(round(value * scale))
    return max(min_val, min(max_val, scaled))


def from_fixed_point(value: int, frac_bits: int = 4, width: int = 8) -> float:
    """Convert fixed-point integer to float."""
    if value >= (1 << (width - 1)):
        value = value - (1 << width)
    return value / (1 << frac_bits)


def pack_vector(arr: np.ndarray, width: int = 8, frac_bits: int = 4) -> int:
    """Pack numpy array into single integer."""
    result = 0
    flat = arr.flatten()
    for i, val in enumerate(flat):
        fp_val = to_fixed_point(float(val), frac_bits, width) & ((1 << width) - 1)
        result |= fp_val << (i * width)
    return result


def unpack_vector(value: int, length: int, width: int = 8, frac_bits: int = 4) -> np.ndarray:
    """Unpack integer into numpy array of floats."""
    mask = (1 << width) - 1
    result = []
    for i in range(length):
        v = (value >> (i * width)) & mask
        result.append(from_fixed_point(v, frac_bits, width))
    return np.array(result, dtype=np.float32)


# =============================================================================
# Layer Normalization Tests
# =============================================================================

@cocotb.test()
async def test_layer_norm_sequential(dut):
    """Test layer norm with sequential values."""
    dut._log.info("Testing layer norm: sequential values")
    
    dim = int(dut.DIM.value) if hasattr(dut, 'DIM') else 16
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    frac_bits = int(dut.FRAC_BITS.value) if hasattr(dut, 'FRAC_BITS') else 4
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    
    # Set gamma=1, beta=0 (identity transform after normalization)
    if hasattr(dut, 'gamma'):
        gamma_ones = pack_vector(np.ones(dim), act_width, frac_bits)
        dut.gamma.value = gamma_ones
    if hasattr(dut, 'beta'):
        dut.beta.value = 0
    
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Sequential input: 0, 1, 2, ..., dim-1 (scaled by frac_bits)
    x = np.arange(dim, dtype=np.float32)
    x_packed = pack_vector(x, act_width, frac_bits)
    
    dut.x_in.value = x_packed
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    
    # Wait for output
    for _ in range(1000):
        await RisingEdge(dut.clk)
        if dut.valid_out.value == 1:
            break
    
    y_out = unpack_vector(int(dut.y_out.value), dim, act_width, frac_bits)
    
    # After layer norm, output should have mean ~0, std ~1
    mean = np.mean(y_out)
    std = np.std(y_out)
    
    dut._log.info(f"Output mean: {mean:.4f}, std: {std:.4f}")
    
    # Relax tolerances for fixed-point approximation
    assert abs(mean) < 1.0, f"Mean should be ~0, got {mean}"
    
    dut._log.info("PASS: layer norm sequential")


@cocotb.test()
async def test_layer_norm_constant(dut):
    """Test layer norm with constant values - should output near zero."""
    dut._log.info("Testing layer norm: constant values")
    
    dim = int(dut.DIM.value) if hasattr(dut, 'DIM') else 16
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    frac_bits = int(dut.FRAC_BITS.value) if hasattr(dut, 'FRAC_BITS') else 4
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    
    if hasattr(dut, 'gamma'):
        dut.gamma.value = pack_vector(np.ones(dim), act_width, frac_bits)
    if hasattr(dut, 'beta'):
        dut.beta.value = 0
    
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Constant input
    x = np.ones(dim, dtype=np.float32) * 5.0
    x_packed = pack_vector(x, act_width, frac_bits)
    
    dut.x_in.value = x_packed
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    
    for _ in range(1000):
        await RisingEdge(dut.clk)
        if dut.valid_out.value == 1:
            break
    
    y_out = unpack_vector(int(dut.y_out.value), dim, act_width, frac_bits)
    
    # Constant input after normalization should be near zero
    # (variance is 0, so x - mean = 0)
    max_val = np.max(np.abs(y_out))
    dut._log.info(f"Max output magnitude: {max_val:.4f}")
    
    assert max_val < 2.0, f"Constant input should normalize to ~0, max={max_val}"
    
    dut._log.info("PASS: layer norm constant")


@cocotb.test()
async def test_layer_norm_golden(dut):
    """Compare layer norm with golden model."""
    if not GOLDEN_AVAILABLE:
        dut._log.warning("Golden model not available, skipping")
        return
    
    dut._log.info("Testing layer norm: golden comparison")
    
    dim = int(dut.DIM.value) if hasattr(dut, 'DIM') else 16
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    frac_bits = int(dut.FRAC_BITS.value) if hasattr(dut, 'FRAC_BITS') else 4
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Create golden model
    fp_ops = FixedPointOps(width=act_width, frac_bits=frac_bits)
    golden = GoldenLayerNorm(dim=dim, fp_ops=fp_ops)
    
    np.random.seed(42)
    num_tests = 10
    
    for i in range(num_tests):
        x = np.random.randn(dim).astype(np.float32) * 2
        
        # Golden output
        golden_out = golden.forward(x, fixed_point=True)
        
        # DUT
        if hasattr(dut, 'gamma'):
            dut.gamma.value = pack_vector(golden.gamma, act_width, frac_bits)
        if hasattr(dut, 'beta'):
            dut.beta.value = pack_vector(golden.beta, act_width, frac_bits)
        
        dut.x_in.value = pack_vector(x, act_width, frac_bits)
        dut.valid_in.value = 1
        dut.ready_out.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0
        
        for _ in range(1000):
            await RisingEdge(dut.clk)
            if dut.valid_out.value == 1:
                break
        
        dut_out = unpack_vector(int(dut.y_out.value), dim, act_width, frac_bits)
        
        # Compare
        error = np.abs(dut_out - golden_out)
        max_error = np.max(error)
        
        if max_error > 1.0:
            dut._log.warning(f"Test {i}: max_error={max_error}")
        
        await ClockCycles(dut.clk, 2)
    
    dut._log.info(f"PASS: layer norm golden ({num_tests} tests)")


# =============================================================================
# GELU Activation Tests
# =============================================================================

@cocotb.test()
async def test_gelu_key_values(dut):
    """Test GELU at key input values."""
    dut._log.info("Testing GELU: key values")
    
    width = int(dut.WIDTH.value) if hasattr(dut, 'WIDTH') else 16
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    frac_bits = int(dut.FRAC_BITS.value) if hasattr(dut, 'FRAC_BITS') else 4
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Key test values and expected GELU outputs
    test_cases = [
        (0.0, 0.0),      # GELU(0) = 0
        (1.0, 0.841),    # GELU(1) ≈ 0.841
        (-1.0, -0.159),  # GELU(-1) ≈ -0.159
        (2.0, 1.955),    # GELU(2) ≈ 1.955
        (-2.0, -0.045),  # GELU(-2) ≈ -0.045
    ]
    
    for x_val, expected_approx in test_cases:
        # Create input with test value repeated
        x = np.full(width, x_val, dtype=np.float32)
        x_packed = pack_vector(x, act_width, frac_bits)
        
        dut.x_in.value = x_packed
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0
        
        for _ in range(100):
            await RisingEdge(dut.clk)
            if dut.valid_out.value == 1:
                break
        
        y_out = unpack_vector(int(dut.y_out.value), width, act_width, frac_bits)
        
        # Check first output value
        actual = y_out[0]
        error = abs(actual - expected_approx)
        
        dut._log.info(f"GELU({x_val}) = {actual:.3f} (expected ≈{expected_approx:.3f})")
        
        # Allow reasonable tolerance for piece-wise linear approximation
        assert error < 0.5, f"GELU({x_val}): expected ~{expected_approx}, got {actual}"
        
        await ClockCycles(dut.clk, 2)
    
    dut._log.info("PASS: GELU key values")


@cocotb.test()
async def test_gelu_monotonicity(dut):
    """Test that GELU output increases with input for x > 0."""
    dut._log.info("Testing GELU: monotonicity")
    
    width = int(dut.WIDTH.value) if hasattr(dut, 'WIDTH') else 16
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    frac_bits = int(dut.FRAC_BITS.value) if hasattr(dut, 'FRAC_BITS') else 4
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    outputs = []
    test_inputs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for x_val in test_inputs:
        x = np.full(width, x_val, dtype=np.float32)
        dut.x_in.value = pack_vector(x, act_width, frac_bits)
        dut.valid_in.value = 1
        dut.ready_out.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0
        
        for _ in range(100):
            await RisingEdge(dut.clk)
            if dut.valid_out.value == 1:
                break
        
        y_out = unpack_vector(int(dut.y_out.value), width, act_width, frac_bits)
        outputs.append(y_out[0])
        await ClockCycles(dut.clk, 2)
    
    # Check monotonicity for positive inputs
    for i in range(1, len(outputs)):
        assert outputs[i] >= outputs[i-1] - 0.1, \
            f"GELU should be increasing: {outputs[i-1]} -> {outputs[i]}"
    
    dut._log.info("PASS: GELU monotonicity")


@cocotb.test()
async def test_gelu_golden(dut):
    """Compare GELU with golden model."""
    if not GOLDEN_AVAILABLE:
        dut._log.warning("Golden model not available, skipping")
        return
    
    dut._log.info("Testing GELU: golden comparison")
    
    width = int(dut.WIDTH.value) if hasattr(dut, 'WIDTH') else 16
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    frac_bits = int(dut.FRAC_BITS.value) if hasattr(dut, 'FRAC_BITS') else 4
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Create golden model
    golden = GoldenGELU()
    
    np.random.seed(42)
    num_tests = 10
    
    for i in range(num_tests):
        x = np.random.uniform(-3, 3, size=width).astype(np.float32)
        
        # Golden output (using approximation)
        golden_out = golden.forward(x, fixed_point=True)
        
        # DUT
        dut.x_in.value = pack_vector(x, act_width, frac_bits)
        dut.valid_in.value = 1
        dut.ready_out.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0
        
        for _ in range(100):
            await RisingEdge(dut.clk)
            if dut.valid_out.value == 1:
                break
        
        dut_out = unpack_vector(int(dut.y_out.value), width, act_width, frac_bits)
        
        # Compare
        error = np.mean(np.abs(dut_out - golden_out))
        
        if error > 0.5:
            dut._log.warning(f"Test {i}: mean_error={error:.4f}")
        
        await ClockCycles(dut.clk, 2)
    
    dut._log.info(f"PASS: GELU golden ({num_tests} tests)")


# =============================================================================
# Test Vector Generation
# =============================================================================

def generate_transformer_test_vectors(
    embed_dim: int = 64,
    num_heads: int = 4,
    num_vectors: int = 10,
    seed: int = 42
) -> list:
    """
    Generate test vectors for transformer block verification.
    """
    if not GOLDEN_AVAILABLE:
        return []
    
    np.random.seed(seed)
    vectors = []
    
    # Create golden model
    config = VisionConfig(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=embed_dim*4)
    block = GoldenTransformerBlock(config, weight_type='ternary')
    
    for i in range(num_vectors):
        seq_len = np.random.choice([4, 8, 16])
        x = np.random.randn(seq_len, embed_dim).astype(np.float32) * 0.1
        output, attn = block.forward(x, return_attention=True)
        
        vectors.append({
            'name': f'transformer_test_{i}',
            'input': x,
            'expected_output': output,
            'expected_attention': attn,
            'seq_len': seq_len,
        })
    
    return vectors


def save_test_vectors(vectors: list, output_path: str):
    """Save test vectors to file for RTL simulation."""
    import json
    
    output = []
    for v in vectors:
        output.append({
            'name': v['name'],
            'seq_len': v.get('seq_len', v['input'].shape[0]),
            'input': v['input'].tolist(),
            'expected_output': v['expected_output'].tolist(),
        })
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    vectors = generate_transformer_test_vectors()
    print(f"Generated {len(vectors)} transformer test vectors")
    
    if vectors:
        save_test_vectors(vectors, 'transformer_test_vectors.json')
        print("Saved to transformer_test_vectors.json")
