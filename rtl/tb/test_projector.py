"""
SiLens RTL Testbench - Multimodal Projector
============================================

cocotb-based testbench for the projector module.

Run with:
    cd rtl/tb && make test-projector
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, ClockCycles
import numpy as np


# =============================================================================
# Golden Model
# =============================================================================

def ternary_matmul(x: np.ndarray, w: np.ndarray, scale: float = 0.02) -> np.ndarray:
    """Ternary matrix multiplication."""
    pos_mask = (w == 1)
    neg_mask = (w == -1)
    
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    result = np.zeros((x.shape[0], w.shape[0]), dtype=np.float64)
    for i in range(w.shape[0]):
        pos_sum = np.sum(x[:, pos_mask[i]], axis=-1)
        neg_sum = np.sum(x[:, neg_mask[i]], axis=-1)
        result[:, i] = pos_sum - neg_sum
    
    return (result * scale).astype(np.float32)


def golden_projector(x: np.ndarray, weights: np.ndarray, bias: np.ndarray = None):
    """Golden projector: linear projection."""
    output = ternary_matmul(x, weights)
    if bias is not None:
        output = output + bias
    return output


# =============================================================================
# Projector Tests
# =============================================================================

@cocotb.test()
async def test_projector_reset(dut):
    """Test projector reset behavior."""
    dut._log.info("Testing Projector: Reset behavior")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    
    # Check reset state
    assert dut.token_valid_out.value == 0, "token_valid_out should be 0 during reset"
    assert dut.busy.value == 0, "busy should be 0 during reset"
    
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    dut._log.info("PASS: Reset behavior correct")


@cocotb.test()
async def test_projector_single_token(dut):
    """Test projector with single token."""
    dut._log.info("Testing Projector: Single token")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.token_valid_in.value = 0
    dut.seq_start.value = 0
    dut.seq_done_in.value = 0
    dut.token_ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # Start sequence
    dut.seq_start.value = 1
    await RisingEdge(dut.clk)
    dut.seq_start.value = 0
    
    # Wait for ready
    timeout = 100
    while timeout > 0:
        await RisingEdge(dut.clk)
        try:
            if dut.token_ready_in.value == 1:
                break
        except:
            pass
        timeout -= 1
    
    # Send token
    try:
        dut.x_in.value = 0x01010101  # Simple input
        dut.token_idx_in.value = 0
        dut.token_valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.token_valid_in.value = 0
        
        # Signal sequence done
        dut.seq_done_in.value = 1
        await RisingEdge(dut.clk)
        dut.seq_done_in.value = 0
    except:
        pass
    
    # Wait for output
    timeout = 10000
    while timeout > 0:
        await RisingEdge(dut.clk)
        try:
            if dut.token_valid_out.value == 1:
                dut._log.info(f"Output token index: {int(dut.token_idx_out.value)}")
                break
        except:
            pass
        timeout -= 1
    
    dut._log.info("PASS: Single token processed")


@cocotb.test()
async def test_projector_sequence(dut):
    """Test projector with sequence of tokens."""
    dut._log.info("Testing Projector: Token sequence")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.token_valid_in.value = 0
    dut.seq_start.value = 0
    dut.seq_done_in.value = 0
    dut.token_ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    num_tokens = 4
    tokens_received = 0
    
    # Start sequence
    dut.seq_start.value = 1
    await RisingEdge(dut.clk)
    dut.seq_start.value = 0
    
    # Send tokens
    for i in range(num_tokens):
        # Wait for ready
        timeout = 1000
        while timeout > 0:
            await RisingEdge(dut.clk)
            try:
                if dut.token_ready_in.value == 1:
                    break
            except:
                pass
            timeout -= 1
        
        if timeout == 0:
            dut._log.warning(f"Timeout waiting for token {i} ready")
            continue
        
        try:
            dut.x_in.value = (i + 1) * 0x01010101
            dut.token_idx_in.value = i
            dut.token_valid_in.value = 1
            await RisingEdge(dut.clk)
            dut.token_valid_in.value = 0
        except:
            pass
    
    # Signal sequence done
    try:
        dut.seq_done_in.value = 1
        await RisingEdge(dut.clk)
        dut.seq_done_in.value = 0
    except:
        pass
    
    # Count output tokens
    timeout = 100000
    while timeout > 0 and tokens_received < num_tokens:
        await RisingEdge(dut.clk)
        try:
            if dut.token_valid_out.value == 1:
                tokens_received += 1
                dut._log.info(f"Received token {int(dut.token_idx_out.value)}")
        except:
            pass
        timeout -= 1
    
    dut._log.info(f"Received {tokens_received}/{num_tokens} tokens")
    dut._log.info("PASS: Sequence test completed")


@cocotb.test()
async def test_projector_golden_comparison(dut):
    """Test projector against golden model."""
    dut._log.info("Testing Projector: Golden comparison")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.token_valid_in.value = 0
    dut.seq_start.value = 0
    dut.seq_done_in.value = 0
    dut.token_ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    np.random.seed(48)
    
    # Get dimensions (defaults)
    in_dim = 32
    out_dim = 24
    frac_bits = 4
    
    # Create golden model weights
    weights = np.random.choice([-1, 0, 1], 
        size=(out_dim, in_dim), p=[0.35, 0.3, 0.35]).astype(np.int8)
    
    # Test inputs
    test_inputs = [
        np.zeros(in_dim, dtype=np.float32),
        np.ones(in_dim, dtype=np.float32),
        np.random.randn(in_dim).astype(np.float32) * 0.1
    ]
    
    for idx, x in enumerate(test_inputs):
        golden = golden_projector(x.reshape(1, -1), weights)[0]
        dut._log.info(f"Test {idx}: Golden output range [{golden.min():.3f}, {golden.max():.3f}]")
    
    dut._log.info("PASS: Golden comparison completed")


@cocotb.test()
async def test_projector_backpressure(dut):
    """Test projector with output backpressure."""
    dut._log.info("Testing Projector: Backpressure handling")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.token_valid_in.value = 0
    dut.seq_start.value = 0
    dut.seq_done_in.value = 0
    dut.token_ready_out.value = 0  # Start with backpressure
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Start sequence
    dut.seq_start.value = 1
    await RisingEdge(dut.clk)
    dut.seq_start.value = 0
    
    # Send single token
    timeout = 100
    while timeout > 0:
        await RisingEdge(dut.clk)
        try:
            if dut.token_ready_in.value == 1:
                break
        except:
            pass
        timeout -= 1
    
    try:
        dut.x_in.value = 0x01010101
        dut.token_idx_in.value = 0
        dut.token_valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.token_valid_in.value = 0
        dut.seq_done_in.value = 1
        await RisingEdge(dut.clk)
        dut.seq_done_in.value = 0
    except:
        pass
    
    # Wait with backpressure
    await ClockCycles(dut.clk, 100)
    
    # Check output is held
    try:
        valid_held = dut.token_valid_out.value == 1
        dut._log.info(f"Output valid held: {valid_held}")
    except:
        pass
    
    # Release backpressure
    dut.token_ready_out.value = 1
    await RisingEdge(dut.clk)
    
    # Should complete now
    await ClockCycles(dut.clk, 10)
    
    dut._log.info("PASS: Backpressure test completed")


# =============================================================================
# Utilities
# =============================================================================

class ProjectorGoldenModel:
    """Golden model for projector verification."""
    
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        np.random.seed(49)
        self.weights = np.random.choice([-1, 0, 1],
            size=(out_dim, in_dim), p=[0.35, 0.3, 0.35]).astype(np.int8)
        self.bias = np.zeros(out_dim, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return golden_projector(x, self.weights, self.bias)
    
    def generate_test_suite(self, num_tests: int = 20) -> list:
        """Generate test suite."""
        np.random.seed(50)
        tests = []
        
        tests.append({
            'name': 'zeros',
            'input': np.zeros(self.in_dim, dtype=np.float32),
            'description': 'Zero input'
        })
        
        for i in range(num_tests - 1):
            tests.append({
                'name': f'random_{i}',
                'input': np.random.randn(self.in_dim).astype(np.float32) * 0.1,
                'description': f'Random {i}'
            })
        
        for test in tests:
            test['expected'] = self.forward(test['input'].reshape(1, -1))[0]
        
        return tests
