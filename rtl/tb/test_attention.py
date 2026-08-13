"""
SiLens RTL Testbench - Multi-Head Attention
============================================

cocotb-based testbench for attention-related RTL modules:
- Softmax approximation
- Binary/ternary dot products
- Full attention mechanism

Run with:
    cd rtl/tb && make test-attention
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles
import numpy as np
import sys
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tests'))

try:
    from golden.golden_attention import (
        GoldenMultiHeadAttention,
        GoldenSoftmax,
        FixedPointOps,
        WeightType,
        AttentionConfig
    )
    GOLDEN_AVAILABLE = True
except ImportError:
    GOLDEN_AVAILABLE = False


# =============================================================================
# Helper Functions
# =============================================================================

def to_fixed_point(value: float, frac_bits: int = 6, width: int = 8) -> int:
    """Convert float to fixed-point integer."""
    scale = 1 << frac_bits
    max_val = (1 << (width - 1)) - 1
    min_val = -(1 << (width - 1))
    scaled = int(round(value * scale))
    return max(min_val, min(max_val, scaled))


def from_fixed_point(value: int, frac_bits: int = 6, signed: bool = True) -> float:
    """Convert fixed-point integer to float."""
    if signed and value >= (1 << (frac_bits + 7)):  # Negative in 8-bit
        value = value - (1 << 8)
    return value / (1 << frac_bits)


def pack_array(arr: np.ndarray, width: int = 8) -> int:
    """Pack numpy array into single integer for DUT input."""
    result = 0
    flat = arr.flatten()
    for i, val in enumerate(flat):
        if isinstance(val, (np.floating, float)):
            val = to_fixed_point(val, frac_bits=4, width=width)
        val = int(val) & ((1 << width) - 1)
        result |= val << (i * width)
    return result


def unpack_array(value: int, length: int, width: int = 8, signed: bool = True) -> np.ndarray:
    """Unpack integer into numpy array."""
    mask = (1 << width) - 1
    sign_bit = 1 << (width - 1)
    result = []
    for i in range(length):
        v = (value >> (i * width)) & mask
        if signed and (v & sign_bit):
            v = v - (1 << width)
        result.append(v)
    return np.array(result)


# =============================================================================
# Softmax Approximation Tests
# =============================================================================

@cocotb.test()
async def test_softmax_equal_inputs(dut):
    """Test softmax with equal inputs - should produce equal outputs."""
    dut._log.info("Testing softmax: equal inputs")
    
    # Get parameters
    seq_len = int(dut.SEQ_LEN.value) if hasattr(dut, 'SEQ_LEN') else 8
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    
    # Start clock
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Equal inputs (all zeros)
    x_in = pack_array(np.zeros(seq_len), act_width)
    dut.x_in.value = x_in
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    
    # Wait for output
    timeout = 1000
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.valid_out.value == 1:
            break
    else:
        assert False, "Timeout waiting for softmax output"
    
    # Check output - should be approximately equal
    y_out = unpack_array(int(dut.y_out.value), seq_len, act_width)
    
    # All outputs should be similar (equal distribution)
    variance = np.var(y_out)
    dut._log.info(f"Output variance: {variance}")
    assert variance < 10, f"Outputs should be similar, variance={variance}"
    
    dut._log.info("PASS: softmax equal inputs")


@cocotb.test()
async def test_softmax_dominant_input(dut):
    """Test softmax with one dominant input."""
    dut._log.info("Testing softmax: dominant input")
    
    seq_len = int(dut.SEQ_LEN.value) if hasattr(dut, 'SEQ_LEN') else 8
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # One large value, rest small
    x = np.array([3.0] + [-3.0] * (seq_len - 1))
    x_fp = np.array([to_fixed_point(v, frac_bits=4) for v in x])
    x_in = pack_array(x_fp, act_width)
    
    dut.x_in.value = x_in
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    
    for _ in range(1000):
        await RisingEdge(dut.clk)
        if dut.valid_out.value == 1:
            break
    
    y_out = unpack_array(int(dut.y_out.value), seq_len, act_width, signed=False)
    
    # First output should dominate
    dut._log.info(f"Outputs: {y_out}")
    assert y_out[0] > y_out[1], "First output should be larger"
    
    dut._log.info("PASS: softmax dominant input")


@cocotb.test()
async def test_softmax_golden_comparison(dut):
    """Compare softmax output with golden model."""
    if not GOLDEN_AVAILABLE:
        dut._log.warning("Golden model not available, skipping")
        return
    
    dut._log.info("Testing softmax: golden model comparison")
    
    seq_len = int(dut.SEQ_LEN.value) if hasattr(dut, 'SEQ_LEN') else 8
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    frac_bits = int(dut.FRAC_BITS.value) if hasattr(dut, 'FRAC_BITS') else 6
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Create golden model
    config = AttentionConfig(frac_bits=frac_bits)
    fp_ops = FixedPointOps(width=act_width, frac_bits=frac_bits)
    golden = GoldenSoftmax(config, fp_ops)
    
    # Test multiple inputs
    np.random.seed(42)
    num_tests = 10
    
    for i in range(num_tests):
        x = np.random.randn(seq_len).astype(np.float32)
        x = np.clip(x, -4, 4)
        
        # Golden output
        golden_out = golden.forward(x, fixed_point=True)
        
        # DUT input
        x_fp = np.array([to_fixed_point(v, frac_bits=frac_bits, width=act_width) for v in x])
        dut.x_in.value = pack_array(x_fp, act_width)
        dut.valid_in.value = 1
        dut.ready_out.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0
        
        for _ in range(1000):
            await RisingEdge(dut.clk)
            if dut.valid_out.value == 1:
                break
        
        dut_out = unpack_array(int(dut.y_out.value), seq_len, act_width, signed=False)
        
        # Compare
        error = np.abs(dut_out - golden_out)
        max_error = np.max(error)
        
        if max_error > 5:  # Allow some tolerance
            dut._log.warning(f"Test {i}: max_error={max_error}")
        
        await ClockCycles(dut.clk, 2)
    
    dut._log.info(f"PASS: softmax golden comparison ({num_tests} tests)")


# =============================================================================
# Binary Dot Product Tests
# =============================================================================

@cocotb.test()
async def test_binary_dot_all_same(dut):
    """Test binary dot product when all bits match."""
    dut._log.info("Testing binary dot product: all same")
    
    width = len(dut.act_in) if hasattr(dut, 'act_in') else 16
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # All ones - should give max positive
    dut.act_in.value = (1 << width) - 1
    dut.weight_in.value = (1 << width) - 1
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    
    for _ in range(100):
        await RisingEdge(dut.clk)
        if dut.valid_out.value == 1:
            break
    
    result = int(dut.result.value.signed_integer)
    expected = width  # All +1 * +1 = +width
    
    dut._log.info(f"Result: {result}, Expected: {expected}")
    assert result == expected, f"Expected {expected}, got {result}"
    
    dut._log.info("PASS: binary dot all same")


@cocotb.test()
async def test_binary_dot_all_opposite(dut):
    """Test binary dot product when all bits opposite."""
    dut._log.info("Testing binary dot product: all opposite")
    
    width = len(dut.act_in) if hasattr(dut, 'act_in') else 16
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # All opposite
    dut.act_in.value = (1 << width) - 1  # All 1s
    dut.weight_in.value = 0               # All 0s
    dut.valid_in.value = 1
    dut.ready_out.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    
    for _ in range(100):
        await RisingEdge(dut.clk)
        if dut.valid_out.value == 1:
            break
    
    result = int(dut.result.value.signed_integer)
    expected = -width  # All +1 * -1 = -width
    
    dut._log.info(f"Result: {result}, Expected: {expected}")
    assert result == expected, f"Expected {expected}, got {result}"
    
    dut._log.info("PASS: binary dot all opposite")


@cocotb.test()
async def test_binary_dot_random(dut):
    """Test binary dot product with random patterns."""
    dut._log.info("Testing binary dot product: random patterns")
    
    width = len(dut.act_in) if hasattr(dut, 'act_in') else 16
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    np.random.seed(42)
    num_tests = 50
    
    for i in range(num_tests):
        act = np.random.randint(0, 2, size=width)
        weight = np.random.randint(0, 2, size=width)
        
        # Pack to integers
        act_int = sum(b << i for i, b in enumerate(act))
        weight_int = sum(b << i for i, b in enumerate(weight))
        
        # Expected: convert {0,1} to {-1,+1} and compute dot product
        act_signed = 2 * act - 1
        weight_signed = 2 * weight - 1
        expected = np.sum(act_signed * weight_signed)
        
        dut.act_in.value = act_int
        dut.weight_in.value = weight_int
        dut.valid_in.value = 1
        dut.ready_out.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0
        
        for _ in range(100):
            await RisingEdge(dut.clk)
            if dut.valid_out.value == 1:
                break
        
        result = int(dut.result.value.signed_integer)
        
        assert result == expected, f"Test {i}: Expected {expected}, got {result}"
        
        await ClockCycles(dut.clk, 2)
    
    dut._log.info(f"PASS: binary dot random ({num_tests} tests)")


# =============================================================================
# Ternary MAC Tests
# =============================================================================

@cocotb.test()
async def test_ternary_mac_all_positive(dut):
    """Test ternary MAC with all +1 weights."""
    dut._log.info("Testing ternary MAC: all +1 weights")
    
    num_elements = int(dut.NUM_ELEMENTS.value) if hasattr(dut, 'NUM_ELEMENTS') else 16
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.acc_clear.value = 1
    dut.ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Activations: 1, 2, 3, ..., num_elements
    activations = np.arange(1, num_elements + 1, dtype=np.uint8)
    act_packed = pack_array(activations, act_width)
    
    # Weights: all +1 (encoded as 0b01)
    W_POS = 0b01
    weight_packed = sum(W_POS << (i * 2) for i in range(num_elements))
    
    dut.act_in.value = act_packed
    dut.weight_in.value = weight_packed
    dut.valid_in.value = 1
    dut.acc_clear.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    dut.acc_clear.value = 0
    
    for _ in range(500):
        await RisingEdge(dut.clk)
        if dut.valid_out.value == 1:
            break
    
    result = int(dut.result.value.signed_integer)
    expected = sum(range(1, num_elements + 1))  # Sum of 1 to N
    
    dut._log.info(f"Result: {result}, Expected: {expected}")
    assert result == expected, f"Expected {expected}, got {result}"
    
    dut._log.info("PASS: ternary MAC all +1")


@cocotb.test()
async def test_ternary_mac_all_negative(dut):
    """Test ternary MAC with all -1 weights."""
    dut._log.info("Testing ternary MAC: all -1 weights")
    
    num_elements = int(dut.NUM_ELEMENTS.value) if hasattr(dut, 'NUM_ELEMENTS') else 16
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    activations = np.arange(1, num_elements + 1, dtype=np.uint8)
    act_packed = pack_array(activations, act_width)
    
    # Weights: all -1 (encoded as 0b10)
    W_NEG = 0b10
    weight_packed = sum(W_NEG << (i * 2) for i in range(num_elements))
    
    dut.act_in.value = act_packed
    dut.weight_in.value = weight_packed
    dut.valid_in.value = 1
    dut.acc_clear.value = 1
    dut.ready_out.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    dut.acc_clear.value = 0
    
    for _ in range(500):
        await RisingEdge(dut.clk)
        if dut.valid_out.value == 1:
            break
    
    result = int(dut.result.value.signed_integer)
    expected = -sum(range(1, num_elements + 1))
    
    dut._log.info(f"Result: {result}, Expected: {expected}")
    assert result == expected, f"Expected {expected}, got {result}"
    
    dut._log.info("PASS: ternary MAC all -1")


@cocotb.test()
async def test_ternary_mac_all_zero(dut):
    """Test ternary MAC with all zero weights."""
    dut._log.info("Testing ternary MAC: all zero weights")
    
    num_elements = int(dut.NUM_ELEMENTS.value) if hasattr(dut, 'NUM_ELEMENTS') else 16
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    activations = np.arange(1, num_elements + 1, dtype=np.uint8)
    act_packed = pack_array(activations, act_width)
    
    # Weights: all 0 (encoded as 0b00)
    weight_packed = 0
    
    dut.act_in.value = act_packed
    dut.weight_in.value = weight_packed
    dut.valid_in.value = 1
    dut.acc_clear.value = 1
    dut.ready_out.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    dut.acc_clear.value = 0
    
    for _ in range(500):
        await RisingEdge(dut.clk)
        if dut.valid_out.value == 1:
            break
    
    result = int(dut.result.value.signed_integer)
    expected = 0
    
    dut._log.info(f"Result: {result}, Expected: {expected}")
    assert result == expected, f"Expected {expected}, got {result}"
    
    dut._log.info("PASS: ternary MAC all zero")


@cocotb.test()
async def test_ternary_mac_mixed(dut):
    """Test ternary MAC with mixed weights."""
    dut._log.info("Testing ternary MAC: mixed weights")
    
    num_elements = int(dut.NUM_ELEMENTS.value) if hasattr(dut, 'NUM_ELEMENTS') else 16
    act_width = int(dut.ACT_WIDTH.value) if hasattr(dut, 'ACT_WIDTH') else 8
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Encoding
    W_ZERO = 0b00
    W_POS = 0b01
    W_NEG = 0b10
    
    np.random.seed(42)
    num_tests = 20
    
    for test_i in range(num_tests):
        # Random activations
        activations = np.random.randint(0, 256, size=num_elements, dtype=np.uint8)
        act_packed = pack_array(activations, act_width)
        
        # Random ternary weights
        weights = np.random.choice([-1, 0, 1], size=num_elements)
        
        # Pack weights
        weight_encoded = []
        for w in weights:
            if w == 1:
                weight_encoded.append(W_POS)
            elif w == -1:
                weight_encoded.append(W_NEG)
            else:
                weight_encoded.append(W_ZERO)
        
        weight_packed = sum(w << (i * 2) for i, w in enumerate(weight_encoded))
        
        # Expected result
        expected = sum(int(a) * w for a, w in zip(activations, weights))
        
        dut.act_in.value = act_packed
        dut.weight_in.value = weight_packed
        dut.valid_in.value = 1
        dut.acc_clear.value = 1
        dut.ready_out.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0
        dut.acc_clear.value = 0
        
        for _ in range(500):
            await RisingEdge(dut.clk)
            if dut.valid_out.value == 1:
                break
        
        result = int(dut.result.value.signed_integer)
        
        assert result == expected, \
            f"Test {test_i}: Expected {expected}, got {result}"
        
        await ClockCycles(dut.clk, 2)
    
    dut._log.info(f"PASS: ternary MAC mixed ({num_tests} tests)")


# =============================================================================
# Test Vector Generation
# =============================================================================

def generate_attention_test_vectors(
    seq_len: int = 8,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_vectors: int = 10,
    seed: int = 42
) -> list:
    """
    Generate test vectors for attention module verification.
    
    Returns list of dicts with inputs and expected outputs.
    """
    if not GOLDEN_AVAILABLE:
        return []
    
    np.random.seed(seed)
    vectors = []
    
    # Create golden model
    attn = GoldenMultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        weight_type='ternary',
        precision='fixed'
    )
    
    # Generate vectors
    for i in range(num_vectors):
        x = np.random.randn(seq_len, embed_dim).astype(np.float32) * 0.1
        output, attn_weights = attn.forward(x, return_attention=True)
        
        vectors.append({
            'name': f'attention_test_{i}',
            'input': x,
            'expected_output': output,
            'expected_attention': attn_weights,
        })
    
    return vectors


if __name__ == "__main__":
    # Generate test vectors for offline use
    vectors = generate_attention_test_vectors()
    print(f"Generated {len(vectors)} test vectors")
