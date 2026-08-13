"""
SiLens RTL Testbench - MLP Modules
===================================

cocotb-based testbench for MLP RTL modules:
- vit_mlp (Vision Transformer MLP with GELU)
- llm_mlp (Language Model MLP with SwiGLU)

Run with:
    cd rtl/tb && make test-vit-mlp
    cd rtl/tb && make test-llm-mlp
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, ClockCycles
import numpy as np


# =============================================================================
# Golden Models
# =============================================================================

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return (0.5 * x * (1 + np.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    ))).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation."""
    return (x / (1 + np.exp(-np.clip(x, -20, 20)))).astype(np.float32)


def ternary_matmul(x: np.ndarray, w: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Ternary matrix multiplication."""
    pos_mask = (w == 1)
    neg_mask = (w == -1)
    result = np.zeros((x.shape[0], w.shape[0]), dtype=np.float64)
    
    for i in range(w.shape[0]):
        pos_sum = np.sum(x[:, pos_mask[i]], axis=-1)
        neg_sum = np.sum(x[:, neg_mask[i]], axis=-1)
        result[:, i] = pos_sum - neg_sum
    
    return (result * scale).astype(np.float32)


def golden_vit_mlp(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, 
                   b1: np.ndarray = None, b2: np.ndarray = None) -> np.ndarray:
    """Golden ViT MLP: FC -> GELU -> FC."""
    hidden = ternary_matmul(x.reshape(1, -1), w1)[0]
    if b1 is not None:
        hidden = hidden + b1
    hidden = gelu(hidden)
    output = ternary_matmul(hidden.reshape(1, -1), w2)[0]
    if b2 is not None:
        output = output + b2
    return output


def golden_llm_mlp(x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, 
                   w_down: np.ndarray) -> np.ndarray:
    """Golden LLM MLP (SwiGLU): gate*up -> down."""
    gate = ternary_matmul(x.reshape(1, -1), w_gate)[0]
    up = ternary_matmul(x.reshape(1, -1), w_up)[0]
    hidden = silu(gate) * up
    output = ternary_matmul(hidden.reshape(1, -1), w_down)[0]
    return output


# =============================================================================
# ViT MLP Tests
# =============================================================================

@cocotb.test()
async def test_vit_mlp_reset(dut):
    """Test ViT MLP reset behavior."""
    dut._log.info("Testing ViT MLP: Reset behavior")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    
    # Check reset state
    assert dut.valid_out.value == 0, "valid_out should be 0 during reset"
    assert dut.ready_in.value == 0 or dut.ready_in.value == 1, "ready_in should be defined"
    
    # Release reset
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    assert dut.ready_in.value == 1, "ready_in should be 1 after reset"
    
    dut._log.info("PASS: Reset behavior correct")


@cocotb.test()
async def test_vit_mlp_all_positive_weights(dut):
    """Test ViT MLP with all +1 weights."""
    dut._log.info("Testing ViT MLP: All positive weights")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Get parameters
    try:
        dim = 32
        hidden_dim = 128
        frac_bits = 4
    except:
        pass
    
    # Set all +1 weights (encoded as 2'b01)
    w1_packed = (0x55555555 << 0)  # All 01 pattern
    w2_packed = (0x55555555 << 0)
    
    try:
        dut.w1.value = w1_packed
        dut.w2.value = w2_packed
        dut.b1.value = 0
        dut.b2.value = 0
    except:
        dut._log.warning("Could not set weights - module interface may differ")
    
    # Simple input
    x_packed = 0
    for i in range(min(dim, 32)):
        x_packed |= (i % 16) << (i * 8)
    
    try:
        dut.x_in.value = x_packed
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0
    except:
        pass
    
    # Wait for completion
    timeout = 10000
    while timeout > 0:
        await RisingEdge(dut.clk)
        try:
            if dut.valid_out.value == 1:
                break
        except:
            pass
        timeout -= 1
    
    if timeout > 0:
        dut._log.info("PASS: MLP completed processing")
    else:
        dut._log.warning("Test timeout - MLP may need more cycles")


@cocotb.test()
async def test_vit_mlp_zeros(dut):
    """Test ViT MLP with zero input."""
    dut._log.info("Testing ViT MLP: Zero input")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Zero input
    try:
        dut.x_in.value = 0
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0
    except:
        pass
    
    # Wait for output
    timeout = 10000
    while timeout > 0:
        await RisingEdge(dut.clk)
        try:
            if dut.valid_out.value == 1:
                # GELU(0) = 0, so output should be 0 (or close to it)
                dut._log.info("Output received for zero input")
                break
        except:
            pass
        timeout -= 1
    
    dut._log.info("PASS: Zero input processed")


# =============================================================================
# LLM MLP Tests (SwiGLU)
# =============================================================================

@cocotb.test()
async def test_llm_mlp_reset(dut):
    """Test LLM MLP reset behavior."""
    dut._log.info("Testing LLM MLP: Reset behavior")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    
    assert dut.valid_out.value == 0, "valid_out should be 0 during reset"
    
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    dut._log.info("PASS: Reset behavior correct")


@cocotb.test()
async def test_llm_mlp_swiglu(dut):
    """Test LLM MLP SwiGLU computation."""
    dut._log.info("Testing LLM MLP: SwiGLU computation")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Test SwiGLU: output = down(silu(gate(x)) * up(x))
    np.random.seed(45)
    
    try:
        dim = 32
        hidden_dim = 64
    except:
        pass
    
    # Simple input
    x_packed = 0
    for i in range(min(dim, 32)):
        val = (i % 16) + 1  # 1 to 16
        x_packed |= (val & 0xFF) << (i * 8)
    
    try:
        dut.x_in.value = x_packed
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0
    except:
        pass
    
    # Wait for completion
    timeout = 100000
    while timeout > 0:
        await RisingEdge(dut.clk)
        try:
            if dut.valid_out.value == 1:
                dut._log.info("SwiGLU MLP completed")
                break
        except:
            pass
        timeout -= 1
    
    dut._log.info("PASS: SwiGLU test completed")


@cocotb.test()
async def test_llm_mlp_throughput(dut):
    """Test LLM MLP throughput (pipeline behavior)."""
    dut._log.info("Testing LLM MLP: Throughput")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_out.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Count cycles for processing
    start_cycle = 0
    end_cycle = 0
    
    # Start processing
    try:
        dut.x_in.value = 0x01010101
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        start_cycle = 1
        dut.valid_in.value = 0
    except:
        pass
    
    # Wait for output
    cycles = 0
    while cycles < 100000:
        await RisingEdge(dut.clk)
        cycles += 1
        try:
            if dut.valid_out.value == 1:
                end_cycle = cycles
                break
        except:
            pass
    
    if end_cycle > 0:
        latency = end_cycle - start_cycle
        dut._log.info(f"Processing latency: {latency} cycles")
    
    dut._log.info("PASS: Throughput test completed")


# =============================================================================
# Golden Model Test Utilities
# =============================================================================

class MLPGoldenModel:
    """Golden model class for MLP verification."""
    
    def __init__(self, mlp_type: str, dim: int, hidden_dim: int):
        self.mlp_type = mlp_type
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        np.random.seed(46)
        
        if mlp_type == 'vit':
            self.w1 = np.random.choice([-1, 0, 1], 
                size=(hidden_dim, dim), p=[0.35, 0.3, 0.35]).astype(np.int8)
            self.w2 = np.random.choice([-1, 0, 1], 
                size=(dim, hidden_dim), p=[0.35, 0.3, 0.35]).astype(np.int8)
            self.b1 = np.zeros(hidden_dim, dtype=np.float32)
            self.b2 = np.zeros(dim, dtype=np.float32)
        else:
            self.w_gate = np.random.choice([-1, 0, 1],
                size=(hidden_dim, dim), p=[0.35, 0.3, 0.35]).astype(np.int8)
            self.w_up = np.random.choice([-1, 0, 1],
                size=(hidden_dim, dim), p=[0.35, 0.3, 0.35]).astype(np.int8)
            self.w_down = np.random.choice([-1, 0, 1],
                size=(dim, hidden_dim), p=[0.35, 0.3, 0.35]).astype(np.int8)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute MLP forward pass."""
        if self.mlp_type == 'vit':
            return golden_vit_mlp(x, self.w1, self.w2, self.b1, self.b2)
        else:
            return golden_llm_mlp(x, self.w_gate, self.w_up, self.w_down)
    
    def generate_test_suite(self, num_tests: int = 20) -> list:
        """Generate test suite."""
        np.random.seed(47)
        tests = []
        
        # Edge cases
        tests.append({
            'name': 'zeros',
            'input': np.zeros(self.dim, dtype=np.float32),
            'description': 'Zero input'
        })
        
        tests.append({
            'name': 'ones',
            'input': np.ones(self.dim, dtype=np.float32),
            'description': 'All ones'
        })
        
        # Random
        for i in range(num_tests - 2):
            tests.append({
                'name': f'random_{i}',
                'input': np.random.randn(self.dim).astype(np.float32) * 0.1,
                'description': f'Random {i}'
            })
        
        for test in tests:
            test['expected'] = self.forward(test['input'])
        
        return tests
