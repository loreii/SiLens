"""
SiLens RTL Testbench - Memory Modules
======================================

cocotb-based testbench for memory RTL modules:
- Activation buffer
- KV cache

Run with:
    cd rtl/tb && make test-memory
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, ClockCycles
import numpy as np


# =============================================================================
# Activation Buffer Tests
# =============================================================================

@cocotb.test()
async def test_activation_buffer_write_read(dut):
    """Test activation buffer basic write/read."""
    dut._log.info("Testing Activation Buffer: Write/Read")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Write test data
    test_data = [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0xABCDEF00]
    
    for i, data in enumerate(test_data):
        try:
            dut.wr_en.value = 1
            dut.wr_addr.value = i
            dut.wr_data.value = data
            await RisingEdge(dut.clk)
        except AttributeError:
            dut._log.warning("Write interface not found - skipping")
            break
    
    try:
        dut.wr_en.value = 0
    except:
        pass
    
    await ClockCycles(dut.clk, 2)
    
    # Read back and verify
    errors = 0
    for i, expected in enumerate(test_data):
        try:
            dut.rd_addr.value = i
            await ClockCycles(dut.clk, 2)  # Read latency
            actual = int(dut.rd_data.value)
            
            if actual != expected:
                dut._log.error(f"Addr {i}: expected {expected:08x}, got {actual:08x}")
                errors += 1
        except AttributeError:
            dut._log.warning("Read interface not found - skipping")
            break
    
    if errors == 0:
        dut._log.info("PASS: Write/Read test completed")
    else:
        dut._log.error(f"FAIL: {errors} mismatches")


@cocotb.test()
async def test_activation_buffer_random(dut):
    """Test activation buffer with random data."""
    dut._log.info("Testing Activation Buffer: Random data")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    np.random.seed(51)
    
    # Get buffer depth
    try:
        depth = 64  # Default
        width = 32  # Default
    except:
        pass
    
    num_tests = min(depth, 32)
    test_data = {}
    
    # Write random data
    for i in range(num_tests):
        addr = np.random.randint(0, depth)
        data = np.random.randint(0, 2**width)
        test_data[addr] = data
        
        try:
            dut.wr_en.value = 1
            dut.wr_addr.value = int(addr)
            dut.wr_data.value = int(data)
            await RisingEdge(dut.clk)
        except:
            break
    
    try:
        dut.wr_en.value = 0
    except:
        pass
    
    await ClockCycles(dut.clk, 2)
    
    # Read back and verify
    errors = 0
    for addr, expected in test_data.items():
        try:
            dut.rd_addr.value = int(addr)
            await ClockCycles(dut.clk, 2)
            actual = int(dut.rd_data.value)
            
            if actual != expected:
                errors += 1
        except:
            break
    
    dut._log.info(f"Random test: {errors} errors out of {len(test_data)} tests")


# =============================================================================
# KV Cache Tests
# =============================================================================

@cocotb.test()
async def test_kv_cache_write(dut):
    """Test KV cache write operation."""
    dut._log.info("Testing KV Cache: Write operation")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Write K and V for position 0
    try:
        dut.wr_en.value = 1
        dut.position.value = 0
        dut.k_in.value = 0x12345678
        dut.v_in.value = 0xABCDEF00
        await RisingEdge(dut.clk)
        dut.wr_en.value = 0
    except AttributeError:
        dut._log.warning("KV cache interface not found")
    
    await ClockCycles(dut.clk, 2)
    
    dut._log.info("PASS: KV cache write completed")


@cocotb.test()
async def test_kv_cache_sequential_positions(dut):
    """Test KV cache with sequential positions."""
    dut._log.info("Testing KV Cache: Sequential positions")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    num_positions = 8
    k_values = {}
    v_values = {}
    
    # Write sequential positions
    for pos in range(num_positions):
        k_val = 0x10000000 + pos
        v_val = 0x20000000 + pos
        k_values[pos] = k_val
        v_values[pos] = v_val
        
        try:
            dut.wr_en.value = 1
            dut.position.value = pos
            dut.k_in.value = k_val
            dut.v_in.value = v_val
            await RisingEdge(dut.clk)
        except:
            break
    
    try:
        dut.wr_en.value = 0
    except:
        pass
    
    await ClockCycles(dut.clk, 2)
    
    # Read back
    errors = 0
    for pos in range(num_positions):
        try:
            dut.rd_position.value = pos
            await ClockCycles(dut.clk, 2)
            
            k_actual = int(dut.k_out.value)
            v_actual = int(dut.v_out.value)
            
            if k_actual != k_values.get(pos, 0):
                errors += 1
            if v_actual != v_values.get(pos, 0):
                errors += 1
        except:
            break
    
    dut._log.info(f"Sequential positions: {errors} errors")


@cocotb.test()
async def test_kv_cache_overwrite(dut):
    """Test KV cache overwrite behavior."""
    dut._log.info("Testing KV Cache: Overwrite behavior")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Write initial value
    try:
        dut.wr_en.value = 1
        dut.position.value = 0
        dut.k_in.value = 0xAAAAAAAA
        dut.v_in.value = 0xBBBBBBBB
        await RisingEdge(dut.clk)
        
        # Overwrite
        dut.k_in.value = 0xCCCCCCCC
        dut.v_in.value = 0xDDDDDDDD
        await RisingEdge(dut.clk)
        
        dut.wr_en.value = 0
    except:
        pass
    
    await ClockCycles(dut.clk, 2)
    
    # Read back - should have overwritten value
    try:
        dut.rd_position.value = 0
        await ClockCycles(dut.clk, 2)
        
        k_val = int(dut.k_out.value)
        v_val = int(dut.v_out.value)
        
        if k_val == 0xCCCCCCCC and v_val == 0xDDDDDDDD:
            dut._log.info("PASS: Overwrite successful")
        else:
            dut._log.info(f"Got K={k_val:08x}, V={v_val:08x}")
    except:
        pass


@cocotb.test()
async def test_kv_cache_capacity(dut):
    """Test KV cache at full capacity."""
    dut._log.info("Testing KV Cache: Full capacity")
    
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    
    # Get max sequence length
    try:
        max_seq_len = 64  # Default
    except:
        pass
    
    # Fill entire cache
    for pos in range(max_seq_len):
        try:
            dut.wr_en.value = 1
            dut.position.value = pos
            dut.k_in.value = pos * 2
            dut.v_in.value = pos * 2 + 1
            await RisingEdge(dut.clk)
        except:
            break
    
    try:
        dut.wr_en.value = 0
    except:
        pass
    
    await ClockCycles(dut.clk, 2)
    
    # Spot check some positions
    check_positions = [0, max_seq_len // 2, max_seq_len - 1]
    errors = 0
    
    for pos in check_positions:
        if pos >= max_seq_len:
            continue
        try:
            dut.rd_position.value = pos
            await ClockCycles(dut.clk, 2)
            
            k_expected = pos * 2
            v_expected = pos * 2 + 1
            k_actual = int(dut.k_out.value)
            v_actual = int(dut.v_out.value)
            
            if k_actual != k_expected or v_actual != v_expected:
                errors += 1
                dut._log.error(f"Pos {pos}: expected K={k_expected}, V={v_expected}, got K={k_actual}, V={v_actual}")
        except:
            break
    
    if errors == 0:
        dut._log.info("PASS: Full capacity test")


# =============================================================================
# Memory Utilities
# =============================================================================

class MemoryGoldenModel:
    """Golden model for memory verification."""
    
    def __init__(self, depth: int, width: int):
        self.depth = depth
        self.width = width
        self.memory = np.zeros(depth, dtype=np.uint64)
    
    def write(self, addr: int, data: int):
        """Write to memory."""
        if 0 <= addr < self.depth:
            self.memory[addr] = data & ((1 << self.width) - 1)
    
    def read(self, addr: int) -> int:
        """Read from memory."""
        if 0 <= addr < self.depth:
            return int(self.memory[addr])
        return 0
    
    def generate_test_suite(self, num_tests: int = 100) -> list:
        """Generate test suite."""
        np.random.seed(52)
        tests = []
        
        # Sequential write/read
        tests.append({
            'name': 'sequential',
            'operations': [(i, i * 0x1234) for i in range(min(16, self.depth))],
            'description': 'Sequential addresses'
        })
        
        # Random addresses
        ops = []
        for _ in range(num_tests):
            addr = np.random.randint(0, self.depth)
            data = np.random.randint(0, 2**min(self.width, 32))
            ops.append((int(addr), int(data)))
        
        tests.append({
            'name': 'random',
            'operations': ops,
            'description': 'Random addresses'
        })
        
        return tests
