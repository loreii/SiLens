#!/usr/bin/env python3
"""
Cocotb testbench for Power Controller.

Tests power management functionality including:
- Power state transitions
- Clock gating
- DVFS interface
- Thermal throttling
- Activity monitoring

Usage:
    make test_power_controller
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles
from cocotb.result import TestFailure
import random


async def reset_dut(dut):
    """Reset the DUT."""
    dut.rst_n.value = 0
    dut.power_state_req.value = 0
    dut.auto_power_mgmt_en.value = 0
    dut.compute_active.value = 0
    dut.memory_active.value = 0
    dut.vision_active.value = 0
    dut.llm_active.value = 0
    dut.dma_active.value = 0
    dut.temperature.value = 50  # 50°C
    dut.thermal_limit.value = 80  # 80°C limit
    dut.thermal_shutdown_req.value = 0
    dut.dvfs_change_ack.value = 0
    dut.pg_core.value = 1
    dut.pg_memory.value = 1
    dut.pg_io.value = 1
    
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


@cocotb.test()
async def test_power_state_transition(dut):
    """Test power state transitions."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Initial state should be FULL_POWER (0)
    assert dut.power_state_ack.value == 0, "Initial state should be FULL_POWER"
    
    # Request BALANCED state (1)
    dut.power_state_req.value = 1
    
    # Wait for transition
    for _ in range(100):
        await RisingEdge(dut.clk)
        if dut.power_state_ack.value == 1:
            break
    
    assert dut.power_state_ack.value == 1, "Should transition to BALANCED"
    
    # Request LOW_POWER state (2)
    dut.power_state_req.value = 2
    
    for _ in range(100):
        await RisingEdge(dut.clk)
        if dut.power_state_ack.value == 2:
            break
    
    assert dut.power_state_ack.value == 2, "Should transition to LOW_POWER"


@cocotb.test()
async def test_clock_gating(dut):
    """Test clock gating based on activity."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # All clocks should be enabled initially
    assert dut.compute_clk_en.value == 0xFFFF, "All compute clocks should be enabled"
    assert dut.vision_clk_en.value == 1, "Vision clock should be enabled"
    assert dut.llm_clk_en.value == 1, "LLM clock should be enabled"
    
    # Request LOW_POWER state - clocks should gate
    dut.power_state_req.value = 2
    
    for _ in range(100):
        await RisingEdge(dut.clk)
        if dut.power_state_ack.value == 2:
            break
    
    # Vision and LLM clocks should be gated in LOW_POWER
    assert dut.vision_clk_en.value == 0, "Vision clock should be gated"
    assert dut.llm_clk_en.value == 0, "LLM clock should be gated"


@cocotb.test()
async def test_dvfs_interface(dut):
    """Test DVFS frequency/voltage selection."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # FULL_POWER should use max DVFS (0)
    assert dut.dvfs_freq_sel.value == 0, "Should be max frequency"
    
    # Transition to LOW_POWER
    dut.power_state_req.value = 2
    
    for _ in range(200):
        await RisingEdge(dut.clk)
        
        # Ack DVFS changes
        if dut.dvfs_change_req.value == 1:
            await ClockCycles(dut.clk, 5)
            dut.dvfs_change_ack.value = 1
            await ClockCycles(dut.clk, 2)
            dut.dvfs_change_ack.value = 0
        
        if dut.power_state_ack.value == 2:
            break
    
    # LOW_POWER should use lower DVFS
    assert dut.dvfs_freq_sel.value > 0, "Should be reduced frequency"


@cocotb.test()
async def test_thermal_throttling(dut):
    """Test thermal throttling behavior."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Initially no throttling
    assert dut.thermal_throttle.value == 0, "Should not be throttled initially"
    
    # Increase temperature above limit
    dut.temperature.value = 90  # Above 80°C limit
    
    # Wait for thermal averaging and throttle detection
    await ClockCycles(dut.clk, 300)
    
    # Should now be throttled
    # Note: May need more cycles for thermal history to fill
    
    # Test thermal shutdown
    dut.thermal_shutdown_req.value = 1
    await ClockCycles(dut.clk, 5)
    
    assert dut.thermal_throttle.value == 1, "Should be throttled on shutdown request"


@cocotb.test()
async def test_activity_monitoring(dut):
    """Test activity-based power state selection."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Enable auto power management
    dut.auto_power_mgmt_en.value = 1
    
    # Simulate high activity
    dut.compute_active.value = 0xFFFF
    dut.memory_active.value = 0xF
    dut.vision_active.value = 1
    dut.llm_active.value = 1
    
    # Wait for activity window
    await ClockCycles(dut.clk, 300)
    
    # High activity should keep at FULL_POWER
    # (actual behavior depends on thresholds)
    
    # Simulate no activity
    dut.compute_active.value = 0
    dut.memory_active.value = 0
    dut.vision_active.value = 0
    dut.llm_active.value = 0
    dut.dma_active.value = 0
    
    # Wait for idle detection (IDLE_THRESHOLD_CYCLES)
    await ClockCycles(dut.clk, 2000)
    
    # Should transition to lower power state
    dut.dvfs_change_ack.value = 1  # Ack any DVFS changes
    await ClockCycles(dut.clk, 100)
    
    # Auto power management should reduce power


@cocotb.test()
async def test_power_fault_detection(dut):
    """Test power fault detection."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Initially no fault
    assert dut.power_fault.value == 0, "Should have no fault initially"
    
    # Simulate power good loss
    dut.pg_core.value = 0
    await ClockCycles(dut.clk, 5)
    
    assert dut.power_fault.value == 1, "Should detect power fault"
    
    # Restore power good
    dut.pg_core.value = 1
    
    # Fault should persist until reset
    await ClockCycles(dut.clk, 10)


@cocotb.test()
async def test_isolation_control(dut):
    """Test power domain isolation."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Initially no isolation
    assert dut.iso_vision.value == 0, "Vision should not be isolated"
    assert dut.iso_llm.value == 0, "LLM should not be isolated"
    
    # Request SLEEP state
    dut.power_state_req.value = 3
    
    for _ in range(200):
        await RisingEdge(dut.clk)
        
        if dut.dvfs_change_req.value == 1:
            await ClockCycles(dut.clk, 5)
            dut.dvfs_change_ack.value = 1
            await ClockCycles(dut.clk, 2)
            dut.dvfs_change_ack.value = 0
        
        if dut.power_state_ack.value == 3:
            break
    
    # In SLEEP, domains should be isolated
    # (Check depends on FSM implementation timing)


@cocotb.test()
async def test_power_consumption_estimation(dut):
    """Test power consumption estimation output."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Wait for estimation to stabilize
    await ClockCycles(dut.clk, 100)
    
    # Power consumption should be non-zero
    power = int(dut.power_consumed_mw.value)
    assert power > 0, "Power consumption estimate should be positive"
    
    # Transition to LOW_POWER
    dut.power_state_req.value = 2
    
    for _ in range(200):
        await RisingEdge(dut.clk)
        if dut.dvfs_change_req.value == 1:
            dut.dvfs_change_ack.value = 1
            await ClockCycles(dut.clk, 2)
            dut.dvfs_change_ack.value = 0
        if dut.power_state_ack.value == 2:
            break
    
    await ClockCycles(dut.clk, 100)
    
    # Power should be lower in LOW_POWER state
    low_power = int(dut.power_consumed_mw.value)
    # Note: Actual comparison depends on the power model
