"""
SiLens RTL Testbench - Common Modules
=====================================

cocotb-based testbench for common RTL modules:
- Popcount (population count / bit counting)
- Future: FIFO, register files, etc.

Run with:
    cd rtl/tb && make test-popcount
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge
import random


# =============================================================================
# Popcount Module Tests
# =============================================================================

def python_popcount(value: int, width: int) -> int:
    """
    Golden model: Count the number of 1 bits in value.
    
    Args:
        value: Integer value to count bits in
        width: Bit width of the value
        
    Returns:
        Number of 1 bits
    """
    # Mask to ensure we only count bits within the width
    mask = (1 << width) - 1
    return bin(value & mask).count('1')


@cocotb.test()
async def test_popcount_all_zeros(dut):
    """
    Test popcount with all zeros input.
    
    Expected behavior:
        Input: 0b00000...
        Output: 0
    """
    dut._log.info("Testing popcount: all zeros")
    
    # Get the width parameter from DUT
    # Use getattr for 'in' since it's a Python reserved word
    dut_in = getattr(dut, 'in')
    width = len(dut_in)
    
    # Apply all zeros
    dut_in.value = 0
    
    # Wait for combinational logic to settle
    await Timer(10, units='ns')
    
    # Check result
    expected = 0
    actual = int(dut.count.value)
    
    assert actual == expected, f"All zeros: expected {expected}, got {actual}"
    dut._log.info(f"PASS: all zeros -> count = {actual}")


@cocotb.test()
async def test_popcount_all_ones(dut):
    """
    Test popcount with all ones input.
    
    Expected behavior:
        Input: 0b11111...
        Output: WIDTH
    """
    dut._log.info("Testing popcount: all ones")
    
    # Get the width parameter from DUT
    dut_in = getattr(dut, 'in')
    width = len(dut_in)
    
    # Apply all ones
    dut_in.value = (1 << width) - 1
    
    # Wait for combinational logic to settle
    await Timer(10, units='ns')
    
    # Check result
    expected = width
    actual = int(dut.count.value)
    
    assert actual == expected, f"All ones: expected {expected}, got {actual}"
    dut._log.info(f"PASS: all ones ({width} bits) -> count = {actual}")


@cocotb.test()
async def test_popcount_single_bit(dut):
    """
    Test popcount with single bit set at each position.
    
    Expected behavior:
        Input: 0b000...1...000 (single bit at position i)
        Output: 1
    """
    dut._log.info("Testing popcount: single bit positions")
    
    dut_in = getattr(dut, 'in')
    width = len(dut_in)
    
    for i in range(min(width, 16)):  # Test first 16 positions for speed
        # Set single bit at position i
        dut_in.value = 1 << i
        
        await Timer(10, units='ns')
        
        expected = 1
        actual = int(dut.count.value)
        
        assert actual == expected, f"Single bit at pos {i}: expected {expected}, got {actual}"
    
    dut._log.info(f"PASS: single bit test for {min(width, 16)} positions")


@cocotb.test()
async def test_popcount_alternating(dut):
    """
    Test popcount with alternating bit patterns.
    
    Expected behavior:
        Input: 0b01010101... or 0b10101010...
        Output: WIDTH // 2
    """
    dut._log.info("Testing popcount: alternating patterns")
    
    dut_in = getattr(dut, 'in')
    width = len(dut_in)
    
    # Pattern 0b0101...
    pattern1 = 0
    for i in range(0, width, 2):
        pattern1 |= (1 << i)
    
    dut_in.value = pattern1
    await Timer(10, units='ns')
    
    expected = python_popcount(pattern1, width)
    actual = int(dut.count.value)
    
    assert actual == expected, f"Pattern 0x55...: expected {expected}, got {actual}"
    
    # Pattern 0b1010...
    pattern2 = 0
    for i in range(1, width, 2):
        pattern2 |= (1 << i)
    
    dut_in.value = pattern2
    await Timer(10, units='ns')
    
    expected = python_popcount(pattern2, width)
    actual = int(dut.count.value)
    
    assert actual == expected, f"Pattern 0xAA...: expected {expected}, got {actual}"
    
    dut._log.info(f"PASS: alternating patterns -> {expected} bits each")


@cocotb.test()
async def test_popcount_random(dut):
    """
    Test popcount with random values.
    
    This test generates 100 random values and compares
    the hardware result with the Python golden model.
    """
    dut._log.info("Testing popcount: random values")
    
    dut_in = getattr(dut, 'in')
    width = len(dut_in)
    num_tests = 100
    
    for i in range(num_tests):
        # Generate random value
        test_value = random.randint(0, (1 << width) - 1)
        
        dut_in.value = test_value
        await Timer(10, units='ns')
        
        expected = python_popcount(test_value, width)
        actual = int(dut.count.value)
        
        assert actual == expected, \
            f"Random test {i}: in=0x{test_value:x}, expected {expected}, got {actual}"
    
    dut._log.info(f"PASS: {num_tests} random tests completed")


@cocotb.test()
async def test_popcount_power_of_two_minus_one(dut):
    """
    Test popcount with values that are power-of-2 minus 1.
    
    Expected behavior:
        Input: 2^n - 1 (n consecutive 1s)
        Output: n
    """
    dut._log.info("Testing popcount: power-of-two minus one")
    
    dut_in = getattr(dut, 'in')
    width = len(dut_in)
    
    for n in range(1, min(width + 1, 17)):  # Test up to 16 bits
        test_value = (1 << n) - 1
        
        dut_in.value = test_value
        await Timer(10, units='ns')
        
        expected = n
        actual = int(dut.count.value)
        
        assert actual == expected, f"2^{n}-1 = {test_value}: expected {expected}, got {actual}"
    
    dut._log.info(f"PASS: power-of-two-minus-one tests completed")


@cocotb.test()
async def test_popcount_edge_cases(dut):
    """
    Test popcount edge cases for 1-bit neural network operations.
    
    For binary neural networks, common patterns include:
    - Dense blocks of 1s (active weights)
    - Sparse patterns (pruned weights)
    """
    dut._log.info("Testing popcount: neural network edge cases")
    
    dut_in = getattr(dut, 'in')
    width = len(dut_in)
    
    # Test patterns common in binary neural networks
    test_cases = [
        # (description, value_generator)
        ("25% ones", lambda w: sum(1 << i for i in range(w // 4))),
        ("50% ones (lower)", lambda w: (1 << (w // 2)) - 1),
        ("75% ones", lambda w: (1 << (3 * w // 4)) - 1),
        ("checkerboard", lambda w: sum(1 << i for i in range(0, w, 2))),
    ]
    
    for desc, gen_value in test_cases:
        test_value = gen_value(width)
        
        dut_in.value = test_value
        await Timer(10, units='ns')
        
        expected = python_popcount(test_value, width)
        actual = int(dut.count.value)
        
        assert actual == expected, f"{desc}: expected {expected}, got {actual}"
        dut._log.info(f"  {desc}: {actual} ones")
    
    dut._log.info("PASS: neural network edge cases")


# =============================================================================
# Utility Functions
# =============================================================================

def generate_random_test_vectors(width: int, count: int, seed: int = 42) -> list:
    """
    Generate reproducible random test vectors.
    
    Args:
        width: Bit width of vectors
        count: Number of vectors to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of (input_value, expected_output) tuples
    """
    random.seed(seed)
    vectors = []
    
    for _ in range(count):
        value = random.randint(0, (1 << width) - 1)
        expected = python_popcount(value, width)
        vectors.append((value, expected))
    
    return vectors


def generate_corner_cases(width: int) -> list:
    """
    Generate corner case test vectors.
    
    Args:
        width: Bit width of vectors
        
    Returns:
        List of (input_value, expected_output, description) tuples
    """
    cases = [
        (0, 0, "all_zeros"),
        ((1 << width) - 1, width, "all_ones"),
        (1, 1, "lsb_only"),
        (1 << (width - 1), 1, "msb_only"),
    ]
    
    # Add alternating patterns
    alt_01 = sum(1 << i for i in range(0, width, 2))
    alt_10 = sum(1 << i for i in range(1, width, 2))
    cases.append((alt_01, python_popcount(alt_01, width), "alternating_01"))
    cases.append((alt_10, python_popcount(alt_10, width), "alternating_10"))
    
    return cases


# =============================================================================
# Golden Model Comparison Utilities
# =============================================================================

class PopcountGoldenModel:
    """
    Golden reference model for popcount operations.
    
    This class provides a Python implementation that matches
    the expected hardware behavior exactly.
    """
    
    def __init__(self, width: int):
        self.width = width
        self.mask = (1 << width) - 1
    
    def count(self, value: int) -> int:
        """Count the number of 1 bits."""
        return bin(value & self.mask).count('1')
    
    def verify(self, input_val: int, output_val: int) -> bool:
        """Verify hardware output matches expected."""
        expected = self.count(input_val)
        return output_val == expected
    
    def generate_test_suite(self, num_random: int = 100) -> list:
        """Generate comprehensive test suite."""
        tests = []
        
        # Corner cases
        for val, expected, desc in generate_corner_cases(self.width):
            tests.append({
                'input': val,
                'expected': expected,
                'description': desc
            })
        
        # Random cases
        for val, expected in generate_random_test_vectors(self.width, num_random):
            tests.append({
                'input': val,
                'expected': expected,
                'description': f'random_0x{val:x}'
            })
        
        return tests
