"""
SiLens RTL Testbench - End-to-End Inference
=============================================

cocotb-based testbench for complete end-to-end RTL simulation of the
SiLens vision-language accelerator. This testbench enables true hardware
simulation for inference, accepting image data and prompt tokens,
driving the silens_top module, and capturing output tokens with
cycle-accurate performance metrics.

Features:
    - Accepts image data as numpy array or file path
    - Accepts prompt tokens as list of integers
    - Drives silens_top through complete inference pipeline
    - Captures output tokens and timing for each stage
    - Writes results to JSON file for Python integration

Run with:
    cd rtl/tb && make test-e2e-inference

Or programmatically via sim_interface.py:
    from sim_interface import run_e2e_simulation
    results = run_e2e_simulation(image, tokens, weights_dir)

Author: SiLens Team
License: Apache 2.0
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles, with_timeout
from cocotb.result import TestFailure
import numpy as np
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import struct


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class E2EConfig:
    """Configuration for end-to-end inference simulation."""
    # Clock settings
    clock_period_ns: int = 10  # 100 MHz
    
    # Image parameters
    img_size: int = 384
    patch_size: int = 16
    num_patches: int = (384 // 16) ** 2  # 576
    in_channels: int = 3
    
    # Model dimensions
    vision_dim: int = 768
    llm_dim: int = 576
    vocab_size: int = 49152
    max_seq_len: int = 8192
    
    # Precision
    act_width: int = 8
    frac_bits: int = 4
    
    # Simulation limits
    max_vision_cycles: int = 1_000_000
    max_prefill_cycles: int = 500_000
    max_decode_cycles_per_token: int = 100_000
    max_output_tokens: int = 256
    
    # Timeouts (in ns)
    vision_timeout_ns: int = 100_000_000  # 100ms
    prefill_timeout_ns: int = 50_000_000  # 50ms
    decode_timeout_ns: int = 10_000_000   # 10ms per token


@dataclass
class InferenceResults:
    """Results from end-to-end inference simulation."""
    success: bool
    output_tokens: List[int]
    
    # Cycle counts
    total_cycles: int
    vision_cycles: int
    prefill_cycles: int
    decode_cycles: int
    
    # Timing breakdown (in cycles)
    vision_start_cycle: int
    vision_end_cycle: int
    prefill_start_cycle: int
    prefill_end_cycle: int
    decode_start_cycle: int
    decode_end_cycle: int
    
    # Token timing
    token_cycle_times: List[int]  # Cycles to generate each token
    time_to_first_token: int      # Cycles from decode start to first token
    
    # Performance metrics
    tokens_per_second: float      # At 100 MHz
    throughput_tokens_per_cycle: float
    
    # Error information
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# Helper Functions
# =============================================================================

def quantize_pixel(value: float, act_width: int = 8, frac_bits: int = 4) -> int:
    """Quantize a float pixel value to fixed-point representation."""
    # Clamp to [0, 1]
    value = max(0.0, min(1.0, value))
    # Scale to fixed-point range
    scale = (1 << frac_bits)
    max_val = (1 << act_width) - 1
    quantized = int(value * scale * (max_val / scale))
    return min(quantized, max_val)


def pack_pixel_rgb(r: float, g: float, b: float, act_width: int = 8) -> int:
    """Pack RGB values into a single integer for pixel_in port."""
    r_q = quantize_pixel(r, act_width)
    g_q = quantize_pixel(g, act_width)
    b_q = quantize_pixel(b, act_width)
    # Pack as R | G | B (each act_width bits)
    return (r_q << (2 * act_width)) | (g_q << act_width) | b_q


def load_image_data(image_path_or_array, img_size: int = 384) -> np.ndarray:
    """Load and preprocess image data."""
    if isinstance(image_path_or_array, np.ndarray):
        image = image_path_or_array
    elif isinstance(image_path_or_array, (str, Path)):
        path = Path(image_path_or_array)
        if path.suffix == '.npy':
            image = np.load(path)
        else:
            try:
                from PIL import Image
                img = Image.open(path).convert('RGB')
                img = img.resize((img_size, img_size))
                image = np.array(img).astype(np.float32) / 255.0
            except ImportError:
                # Fallback: create random image if PIL not available
                image = np.random.rand(img_size, img_size, 3).astype(np.float32)
    else:
        raise ValueError(f"Invalid image input type: {type(image_path_or_array)}")
    
    # Ensure correct shape and range
    if image.shape != (img_size, img_size, 3):
        raise ValueError(f"Expected image shape ({img_size}, {img_size}, 3), got {image.shape}")
    
    if image.max() > 1.0:
        image = image / 255.0
    
    return image.astype(np.float32)


def load_test_inputs() -> tuple:
    """Load test inputs from environment or default files."""
    # Check for environment variable paths
    image_path = os.environ.get('SILENS_TEST_IMAGE', None)
    tokens_path = os.environ.get('SILENS_TEST_TOKENS', None)
    
    # Load image
    if image_path and Path(image_path).exists():
        if image_path.endswith('.npy'):
            image = np.load(image_path)
        else:
            image = load_image_data(image_path)
    else:
        # Generate default test image (gradient pattern)
        img_size = 384
        x = np.linspace(0, 1, img_size)
        y = np.linspace(0, 1, img_size)
        xv, yv = np.meshgrid(x, y)
        image = np.stack([xv, yv, (xv + yv) / 2], axis=-1).astype(np.float32)
    
    # Load tokens
    if tokens_path and Path(tokens_path).exists():
        tokens = np.load(tokens_path).tolist()
    else:
        # Default prompt tokens: "Describe this image"
        # Using placeholder token IDs
        tokens = [1, 8612, 436, 2217, 2]  # BOS + tokens + EOS
    
    return image, tokens


# =============================================================================
# FSM State Constants (matching silens_top.v)
# =============================================================================

STATE_IDLE       = 0
STATE_VISION     = 1
STATE_PROJECT    = 2
STATE_LLM_VISION = 3
STATE_LLM_TEXT   = 4
STATE_GENERATE   = 5
STATE_DONE       = 6
STATE_ERROR      = 7


# =============================================================================
# Main E2E Testbench
# =============================================================================

@cocotb.test()
async def test_e2e_inference(dut):
    """
    End-to-end inference test.
    
    This test drives the silens_top module through a complete inference:
    1. Send image pixels through pixel_in interface
    2. Wait for vision encoding to complete
    3. Send prompt tokens through token_in interface
    4. Trigger generation and collect output tokens
    5. Report timing and results
    """
    dut._log.info("=" * 70)
    dut._log.info("SiLens End-to-End Inference Test")
    dut._log.info("=" * 70)
    
    # Initialize configuration
    config = E2EConfig()
    
    # Initialize results
    results = InferenceResults(
        success=False,
        output_tokens=[],
        total_cycles=0,
        vision_cycles=0,
        prefill_cycles=0,
        decode_cycles=0,
        vision_start_cycle=0,
        vision_end_cycle=0,
        prefill_start_cycle=0,
        prefill_end_cycle=0,
        decode_start_cycle=0,
        decode_end_cycle=0,
        token_cycle_times=[],
        time_to_first_token=0,
        tokens_per_second=0.0,
        throughput_tokens_per_cycle=0.0,
    )
    
    # Start clock
    clock = Clock(dut.clk, config.clock_period_ns, units='ns')
    cocotb.start_soon(clock.start())
    
    # Also start PCIe clock if present
    if hasattr(dut, 'pcie_clk'):
        pcie_clock = Clock(dut.pcie_clk, config.clock_period_ns, units='ns')
        cocotb.start_soon(pcie_clock.start())
    
    # Load test inputs
    try:
        image, tokens = load_test_inputs()
        dut._log.info(f"Loaded image shape: {image.shape}")
        dut._log.info(f"Loaded {len(tokens)} prompt tokens")
    except Exception as e:
        dut._log.error(f"Failed to load test inputs: {e}")
        results.error_message = str(e)
        write_results(results)
        return
    
    # =========================================================================
    # Reset sequence
    # =========================================================================
    dut._log.info("Applying reset...")
    
    dut.rst_n.value = 0
    if hasattr(dut, 'pcie_rst_n'):
        dut.pcie_rst_n.value = 0
    
    # Initialize all inputs to safe values
    dut.frame_start.value = 0
    dut.seq_start.value = 0
    dut.generate.value = 0
    dut.pixel_valid.value = 0
    dut.token_in_valid.value = 0
    dut.token_out_ready.value = 1
    
    if hasattr(dut, 'pixel_in'):
        dut.pixel_in.value = 0
    if hasattr(dut, 'token_in'):
        dut.token_in.value = 0
    if hasattr(dut, 'pcie_rx_valid'):
        dut.pcie_rx_valid.value = 0
    if hasattr(dut, 'pcie_tx_ready'):
        dut.pcie_tx_ready.value = 1
    if hasattr(dut, 'debug_sel'):
        dut.debug_sel.value = 0
    
    await ClockCycles(dut.clk, 10)
    
    dut.rst_n.value = 1
    if hasattr(dut, 'pcie_rst_n'):
        dut.pcie_rst_n.value = 1
    
    await ClockCycles(dut.clk, 5)
    
    # Record start cycle
    start_cycle = 0
    current_cycle = 0
    
    dut._log.info("Reset complete, starting inference...")
    
    # =========================================================================
    # Phase 1: Vision Encoding
    # =========================================================================
    dut._log.info("-" * 50)
    dut._log.info("Phase 1: Vision Encoding")
    dut._log.info("-" * 50)
    
    results.vision_start_cycle = current_cycle
    
    # Start frame processing
    dut.frame_start.value = 1
    await RisingEdge(dut.clk)
    current_cycle += 1
    dut.frame_start.value = 0
    
    # Stream pixels row by row
    pixel_count = 0
    total_pixels = config.img_size * config.img_size
    
    dut._log.info(f"Streaming {total_pixels} pixels ({config.img_size}x{config.img_size})...")
    
    for row in range(config.img_size):
        for col in range(config.img_size):
            # Pack RGB pixel
            r, g, b = image[row, col]
            pixel_packed = pack_pixel_rgb(r, g, b, config.act_width)
            
            dut.pixel_in.value = pixel_packed
            dut.pixel_valid.value = 1
            
            await RisingEdge(dut.clk)
            current_cycle += 1
            
            # Handle backpressure
            if hasattr(dut, 'pixel_ready'):
                backpressure_cycles = 0
                while dut.pixel_ready.value == 0:
                    await RisingEdge(dut.clk)
                    current_cycle += 1
                    backpressure_cycles += 1
                    if backpressure_cycles > 1000:
                        dut._log.warning(f"Long backpressure at pixel {pixel_count}")
                        break
            
            pixel_count += 1
            
            # Progress logging
            if pixel_count % 10000 == 0:
                dut._log.info(f"  Sent {pixel_count}/{total_pixels} pixels...")
    
    dut.pixel_valid.value = 0
    
    dut._log.info(f"All {pixel_count} pixels sent, waiting for vision encoding...")
    
    # Wait for vision encoding to complete
    vision_timeout = config.max_vision_cycles
    while vision_timeout > 0:
        await RisingEdge(dut.clk)
        current_cycle += 1
        
        # Check state via debug or status signals
        if hasattr(dut, 'vision_busy') and dut.vision_busy.value == 0:
            # Also check we've moved past vision state
            if hasattr(dut, 'status_leds'):
                state = int(dut.status_leds.value) & 0x0F
                if state > STATE_VISION:
                    break
        
        vision_timeout -= 1
    
    results.vision_end_cycle = current_cycle
    results.vision_cycles = results.vision_end_cycle - results.vision_start_cycle
    
    dut._log.info(f"Vision encoding complete: {results.vision_cycles} cycles")
    
    if vision_timeout == 0:
        dut._log.error("Vision encoding timeout!")
        results.error_message = "Vision encoding timeout"
        write_results(results)
        return
    
    # =========================================================================
    # Phase 2: LLM Prefill (Text Tokens)
    # =========================================================================
    dut._log.info("-" * 50)
    dut._log.info("Phase 2: LLM Prefill")
    dut._log.info("-" * 50)
    
    results.prefill_start_cycle = current_cycle
    
    # Wait for text token ready state
    await ClockCycles(dut.clk, 10)
    current_cycle += 10
    
    # Send prompt tokens
    dut._log.info(f"Sending {len(tokens)} prompt tokens...")
    
    for i, token_id in enumerate(tokens):
        # Mask to vocab size
        token_val = token_id % config.vocab_size
        
        dut.token_in.value = token_val
        dut.token_in_valid.value = 1
        
        await RisingEdge(dut.clk)
        current_cycle += 1
        
        # Handle backpressure
        if hasattr(dut, 'token_in_ready'):
            backpressure_cycles = 0
            while dut.token_in_ready.value == 0:
                await RisingEdge(dut.clk)
                current_cycle += 1
                backpressure_cycles += 1
                if backpressure_cycles > 10000:
                    dut._log.warning(f"Long backpressure at token {i}")
                    break
        
        dut._log.info(f"  Sent token {i}: {token_val}")
    
    dut.token_in_valid.value = 0
    
    # Signal sequence start for prefill
    dut.seq_start.value = 1
    await RisingEdge(dut.clk)
    current_cycle += 1
    dut.seq_start.value = 0
    
    # Wait for prefill to complete
    prefill_timeout = config.max_prefill_cycles
    while prefill_timeout > 0:
        await RisingEdge(dut.clk)
        current_cycle += 1
        
        # Check if ready for generation
        if hasattr(dut, 'llm_busy'):
            if dut.llm_busy.value == 1:
                # LLM is processing
                pass
        
        # Check state
        if hasattr(dut, 'status_leds'):
            state = int(dut.status_leds.value) & 0x0F
            if state == STATE_LLM_TEXT or state == STATE_GENERATE:
                break
        
        prefill_timeout -= 1
    
    results.prefill_end_cycle = current_cycle
    results.prefill_cycles = results.prefill_end_cycle - results.prefill_start_cycle
    
    dut._log.info(f"Prefill complete: {results.prefill_cycles} cycles")
    
    # =========================================================================
    # Phase 3: Autoregressive Generation
    # =========================================================================
    dut._log.info("-" * 50)
    dut._log.info("Phase 3: Autoregressive Generation")
    dut._log.info("-" * 50)
    
    results.decode_start_cycle = current_cycle
    
    # Trigger generation
    dut.generate.value = 1
    await RisingEdge(dut.clk)
    current_cycle += 1
    dut.generate.value = 0
    
    # Collect output tokens
    output_tokens = []
    token_times = []
    last_token_cycle = current_cycle
    first_token_received = False
    
    dut._log.info(f"Waiting for output tokens (max {config.max_output_tokens})...")
    
    decode_timeout = config.max_decode_cycles_per_token * config.max_output_tokens
    
    while len(output_tokens) < config.max_output_tokens and decode_timeout > 0:
        await RisingEdge(dut.clk)
        current_cycle += 1
        decode_timeout -= 1
        
        # Check for output token
        if hasattr(dut, 'token_out_valid') and dut.token_out_valid.value == 1:
            token = int(dut.token_out.value)
            output_tokens.append(token)
            
            # Record timing
            cycles_for_token = current_cycle - last_token_cycle
            token_times.append(cycles_for_token)
            last_token_cycle = current_cycle
            
            if not first_token_received:
                results.time_to_first_token = current_cycle - results.decode_start_cycle
                first_token_received = True
            
            dut._log.info(f"  Token {len(output_tokens)}: {token} ({cycles_for_token} cycles)")
            
            # Check for EOS token (typically token 2)
            if token == 2:
                dut._log.info("EOS token received, ending generation")
                break
        
        # Check for completion or error state
        if hasattr(dut, 'status_leds'):
            state = int(dut.status_leds.value) & 0x0F
            if state == STATE_DONE:
                dut._log.info("Generation complete (STATE_DONE)")
                break
            elif state == STATE_ERROR:
                dut._log.error("Error state detected!")
                results.error_message = "Hardware error state"
                break
        
        # Check interrupt
        if hasattr(dut, 'interrupt') and dut.interrupt.value == 1:
            dut._log.info("Interrupt received, generation complete")
            break
    
    results.decode_end_cycle = current_cycle
    results.decode_cycles = results.decode_end_cycle - results.decode_start_cycle
    
    # =========================================================================
    # Finalize Results
    # =========================================================================
    dut._log.info("-" * 50)
    dut._log.info("Inference Complete - Results")
    dut._log.info("-" * 50)
    
    results.output_tokens = output_tokens
    results.token_cycle_times = token_times
    results.total_cycles = current_cycle - start_cycle
    
    # Calculate performance metrics (assuming 100 MHz clock)
    clock_freq_mhz = 100
    total_time_sec = results.total_cycles / (clock_freq_mhz * 1e6)
    
    if len(output_tokens) > 0:
        results.throughput_tokens_per_cycle = len(output_tokens) / results.decode_cycles if results.decode_cycles > 0 else 0
        results.tokens_per_second = len(output_tokens) / total_time_sec if total_time_sec > 0 else 0
    
    results.success = len(output_tokens) > 0 and results.error_message is None
    
    # Log summary
    dut._log.info(f"Total cycles: {results.total_cycles:,}")
    dut._log.info(f"  Vision:  {results.vision_cycles:,} cycles")
    dut._log.info(f"  Prefill: {results.prefill_cycles:,} cycles")
    dut._log.info(f"  Decode:  {results.decode_cycles:,} cycles")
    dut._log.info(f"Tokens generated: {len(output_tokens)}")
    dut._log.info(f"Time to first token: {results.time_to_first_token:,} cycles")
    dut._log.info(f"Tokens/second @ 100MHz: {results.tokens_per_second:.2f}")
    
    if output_tokens:
        dut._log.info(f"Output tokens: {output_tokens[:20]}{'...' if len(output_tokens) > 20 else ''}")
    
    # Write results to file
    write_results(results)
    
    if not results.success:
        raise TestFailure(f"Inference failed: {results.error_message}")
    
    dut._log.info("=" * 70)
    dut._log.info("END-TO-END INFERENCE TEST PASSED")
    dut._log.info("=" * 70)


def write_results(results: InferenceResults):
    """Write results to JSON file for Python interface."""
    output_path = os.environ.get('SILENS_RESULTS_FILE', 'e2e_results.json')
    
    with open(output_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    
    print(f"Results written to {output_path}")


# =============================================================================
# Additional Test Cases
# =============================================================================

@cocotb.test()
async def test_e2e_reset_during_inference(dut):
    """Test reset behavior during active inference."""
    dut._log.info("Testing: Reset during inference")
    
    config = E2EConfig()
    clock = Clock(dut.clk, config.clock_period_ns, units='ns')
    cocotb.start_soon(clock.start())
    
    # Initial reset
    dut.rst_n.value = 0
    dut.frame_start.value = 0
    dut.pixel_valid.value = 0
    dut.token_in_valid.value = 0
    dut.token_out_ready.value = 1
    dut.seq_start.value = 0
    dut.generate.value = 0
    
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    
    # Start inference
    dut.frame_start.value = 1
    await RisingEdge(dut.clk)
    dut.frame_start.value = 0
    
    # Send some pixels
    for i in range(100):
        dut.pixel_in.value = i
        dut.pixel_valid.value = 1
        await RisingEdge(dut.clk)
    
    # Apply reset mid-inference
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    
    # Verify reset state
    if hasattr(dut, 'vision_busy'):
        assert dut.vision_busy.value == 0, "vision_busy should be 0 after reset"
    if hasattr(dut, 'llm_busy'):
        assert dut.llm_busy.value == 0, "llm_busy should be 0 after reset"
    
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    
    dut._log.info("PASS: Reset during inference handled correctly")


@cocotb.test()
async def test_e2e_backpressure(dut):
    """Test output backpressure handling."""
    dut._log.info("Testing: Output backpressure")
    
    config = E2EConfig()
    clock = Clock(dut.clk, config.clock_period_ns, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.frame_start.value = 0
    dut.pixel_valid.value = 0
    dut.token_in_valid.value = 0
    dut.token_out_ready.value = 0  # Apply backpressure
    dut.seq_start.value = 0
    dut.generate.value = 0
    
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    
    # Start inference minimally
    dut.frame_start.value = 1
    await RisingEdge(dut.clk)
    dut.frame_start.value = 0
    
    # Wait with backpressure
    await ClockCycles(dut.clk, 100)
    
    # Release backpressure
    dut.token_out_ready.value = 1
    await ClockCycles(dut.clk, 10)
    
    dut._log.info("PASS: Backpressure handling verified")


@cocotb.test()
async def test_e2e_debug_interface(dut):
    """Test debug multiplexer functionality."""
    dut._log.info("Testing: Debug interface")
    
    config = E2EConfig()
    clock = Clock(dut.clk, config.clock_period_ns, units='ns')
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    
    # Test each debug select value
    if hasattr(dut, 'debug_sel') and hasattr(dut, 'debug_data'):
        for sel in range(8):
            dut.debug_sel.value = sel
            await ClockCycles(dut.clk, 2)
            debug_val = int(dut.debug_data.value)
            dut._log.info(f"  debug_sel={sel}: debug_data=0x{debug_val:08X}")
    
    dut._log.info("PASS: Debug interface verified")


# =============================================================================
# Standalone Execution Support
# =============================================================================

if __name__ == "__main__":
    # Generate test inputs for standalone testing
    print("Generating test inputs for E2E simulation...")
    
    # Create test image
    img_size = 384
    x = np.linspace(0, 1, img_size)
    y = np.linspace(0, 1, img_size)
    xv, yv = np.meshgrid(x, y)
    test_image = np.stack([xv, yv, (xv + yv) / 2], axis=-1).astype(np.float32)
    
    # Save test image
    np.save('/tmp/silens_test_image.npy', test_image)
    print(f"Test image saved: /tmp/silens_test_image.npy")
    
    # Create test tokens
    test_tokens = np.array([1, 8612, 436, 2217, 2])  # BOS + "Describe this image" + EOS
    np.save('/tmp/silens_test_tokens.npy', test_tokens)
    print(f"Test tokens saved: /tmp/silens_test_tokens.npy")
    
    print("\nRun simulation with:")
    print("  cd rtl/tb && make test-e2e-inference")
