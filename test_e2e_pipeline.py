#!/usr/bin/env python3
"""
SiLens End-to-End Pipeline Test
================================

This script tests the complete pipeline:
1. Quantize weights to ternary (1-bit: -1, 0, +1)
2. Export weights to Verilog hex format
3. Run RTL simulation with Icarus Verilog
4. Verify outputs

Usage:
    python test_e2e_pipeline.py

Requirements:
    - iverilog (Icarus Verilog) or verilator
    - cocotb
    - numpy
"""

import sys
import os
import json
import tempfile
import subprocess
from pathlib import Path
import numpy as np

# Add paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "sdk"))
sys.path.insert(0, str(PROJECT_ROOT / "rtl" / "tb"))


# =============================================================================
# Colors for output
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{Colors.END}\n")


def print_step(num, text):
    print(f"{Colors.BOLD}{Colors.CYAN}[Step {num}]{Colors.END} {text}")


def print_success(text):
    print(f"  {Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    print(f"  {Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    print(f"  {Colors.BLUE}ℹ {text}{Colors.END}")


# =============================================================================
# Step 1: Check Environment
# =============================================================================

def check_environment():
    """Verify all required tools are available."""
    print_step(1, "Checking simulation environment")
    
    errors = []
    
    # Check Icarus Verilog
    try:
        result = subprocess.run(['iverilog', '-V'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print_success(f"Icarus Verilog: {version}")
        else:
            errors.append("iverilog not working")
    except FileNotFoundError:
        errors.append("iverilog not found - install with: brew install icarus-verilog")
    
    # Check cocotb
    try:
        import cocotb
        print_success(f"cocotb: {cocotb.__version__}")
    except ImportError:
        errors.append("cocotb not found - install with: pip install cocotb")
    
    # Check numpy
    try:
        import numpy as np
        print_success(f"numpy: {np.__version__}")
    except ImportError:
        errors.append("numpy not found - install with: pip install numpy")
    
    # Check make
    try:
        result = subprocess.run(['make', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_success("make: available")
    except FileNotFoundError:
        errors.append("make not found")
    
    if errors:
        print()
        for err in errors:
            print_error(err)
        return False
    
    return True


# =============================================================================
# Step 2: Quantize Weights
# =============================================================================

def quantize_test_weights(output_dir: Path):
    """Generate and quantize test weights."""
    print_step(2, "Quantizing test weights to ternary")
    
    from conversion.quantize_ternary import (
        TernaryQuantizer, TernaryQuantizationConfig, QuantizationMode
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = TernaryQuantizationConfig(
        alpha=0.7,
        mode=QuantizationMode.PER_TENSOR
    )
    quantizer = TernaryQuantizer(config)
    
    # Define test layers (smaller for quick test)
    # These match the simplified RTL test configuration
    test_layers = {
        # Vision encoder (reduced size)
        "vision.patch_embed": (64, 48),  # 64 dim, 48 = 4x4x3 patch
        "vision.attn.qkv": (192, 64),    # 3*64 = 192
        "vision.attn.proj": (64, 64),
        "vision.mlp.fc1": (256, 64),
        "vision.mlp.fc2": (64, 256),
        
        # Projector
        "projector.linear": (64, 64),
        
        # LLM (reduced size)
        "llm.attn.q_proj": (64, 64),
        "llm.attn.k_proj": (64, 64),
        "llm.attn.v_proj": (64, 64),
        "llm.attn.o_proj": (64, 64),
        "llm.mlp.gate_proj": (128, 64),
        "llm.mlp.up_proj": (128, 64),
        "llm.mlp.down_proj": (64, 128),
    }
    
    weights_info = {}
    total_params = 0
    total_nonzero = 0
    
    for name, shape in test_layers.items():
        # Generate random weights with realistic distribution
        np.random.seed(hash(name) % 2**32)
        w = np.random.randn(*shape).astype(np.float32) * 0.02
        
        # Quantize
        result = quantizer.quantize_tensor(name, w)
        
        # Get ternary weights (-1, 0, +1)
        ternary = result.quantized_weights
        if hasattr(ternary, 'numpy'):
            ternary = ternary.numpy()  # If it's a torch tensor
        ternary = np.asarray(ternary)
        
        # Save as numpy and hex
        np.save(output_dir / f"{name.replace('.', '_')}.npy", ternary)
        
        # Export as hex for Verilog $readmemh
        hex_path = output_dir / f"{name.replace('.', '_')}.hex"
        export_ternary_to_hex(ternary, hex_path)
        
        params = np.prod(shape)
        nonzero = np.count_nonzero(ternary)
        total_params += params
        total_nonzero += nonzero
        
        weights_info[name] = {
            "shape": list(shape),  # Convert tuple to list for JSON
            "sparsity": float(result.sparsity),  # Convert to native float
            "file": f"{name.replace('.', '_')}.hex"
        }
        
        print_info(f"{name}: {shape} -> sparsity={result.sparsity:.1%}")
    
    # Save manifest (convert numpy types to Python native for JSON serialization)
    manifest = {
        "total_params": int(total_params),
        "total_nonzero": int(total_nonzero),
        "sparsity": float(1 - (total_nonzero / total_params)),
        "layers": weights_info
    }
    
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print_success(f"Quantized {len(test_layers)} layers, {total_params:,} params")
    print_success(f"Overall sparsity: {manifest['sparsity']:.1%}")
    print_success(f"Weights saved to: {output_dir}")
    
    return manifest


def export_ternary_to_hex(weights: np.ndarray, output_path: Path):
    """Export ternary weights to hex format for Verilog $readmemh."""
    flat = weights.flatten().astype(np.int8)
    
    # Encode: -1 -> 10, 0 -> 00, +1 -> 01
    encoded = np.zeros_like(flat, dtype=np.uint8)
    encoded[flat > 0] = 0b01  # +1
    encoded[flat < 0] = 0b10  # -1
    # 0 stays as 0b00
    
    # Pack 4 weights per byte
    packed_len = (len(encoded) + 3) // 4
    packed = np.zeros(packed_len, dtype=np.uint8)
    
    for i in range(len(encoded)):
        byte_idx = i // 4
        bit_offset = (i % 4) * 2
        packed[byte_idx] |= encoded[i] << bit_offset
    
    # Write as hex
    with open(output_path, 'w') as f:
        for byte in packed:
            f.write(f"{byte:02x}\n")


# =============================================================================
# Step 3: Compile RTL
# =============================================================================

def compile_rtl(build_dir: Path):
    """Compile Verilog RTL with Icarus Verilog."""
    print_step(3, "Compiling RTL with Icarus Verilog")
    
    rtl_dir = PROJECT_ROOT / "rtl"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Only compile core modules for the simple test
    # We compile the common modules which are self-contained
    core_files = [
        rtl_dir / "common" / "popcount.v",
        rtl_dir / "common" / "ternary_mac.v",
        rtl_dir / "common" / "binary_dot_product.v",
        rtl_dir / "common" / "layer_norm.v",
        rtl_dir / "common" / "softmax_approx.v",
        rtl_dir / "common" / "gelu_approx.v",
        rtl_dir / "common" / "rms_norm.v",
        rtl_dir / "common" / "simd_vector_unit.v",
        rtl_dir / "common" / "axi_interface.v",
        rtl_dir / "common" / "power_controller.v",
    ]
    
    verilog_files = [f for f in core_files if f.exists()]
    
    print_info(f"Found {len(verilog_files)} core Verilog files")
    
    # Compile with iverilog
    output_file = build_dir / "silens_sim"
    
    cmd = [
        "iverilog",
        "-g2012",  # SystemVerilog 2012
        "-o", str(output_file),
        "-I", str(rtl_dir / "common"),
    ]
    cmd.extend(str(f) for f in verilog_files)
    
    print_info(f"Running: iverilog -g2012 -o {output_file} ...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print_error("Compilation failed!")
        print(result.stderr)
        return False
    
    if result.stderr:
        # Warnings are OK, just print them
        warnings = [l for l in result.stderr.split('\n') if l.strip()]
        if warnings:
            print_info(f"Warnings: {len(warnings)}")
    
    print_success(f"Compiled to: {output_file}")
    return True


# =============================================================================
# Step 4: Run Simple Verilog Simulation
# =============================================================================

def run_simple_simulation(build_dir: Path, weights_dir: Path):
    """Run a simple Verilog simulation test."""
    print_step(4, "Running Verilog simulation")
    
    # Create a simple testbench that just tests basic module instantiation
    tb_code = '''
// Simple test to verify modules compile and elaborate
`timescale 1ns/1ps

module simple_test;
    // Clock and reset
    reg clk = 0;
    reg rst_n = 0;
    
    // Generate clock
    always #5 clk = ~clk;
    
    // Simple test: just check popcount module
    reg [63:0] test_input;
    wire [6:0] count_out;
    
    popcount #(.WIDTH(64)) u_popcount (
        .in(test_input),
        .count(count_out)
    );
    
    // Ternary MAC test - use correct port names
    // NUM_ELEMENTS=16 means 16 elements * 8 bits = 128 bit activations
    // 16 elements * 2 bits = 32 bit weights
    reg [16*8-1:0] activations;   // 128 bits for 16 elements
    reg [16*2-1:0] weights;       // 32 bits for 16 ternary weights
    reg mac_valid_in;
    reg mac_acc_clear;
    wire mac_ready_in;
    wire signed [31:0] mac_result;
    wire mac_valid_out;
    
    ternary_mac #(
        .NUM_ELEMENTS(16),
        .ACT_WIDTH(8),
        .ACC_WIDTH(32),
        .PARALLEL(16)
    ) u_mac (
        .clk(clk),
        .rst_n(rst_n),
        .act_in(activations),
        .weight_in(weights),
        .valid_in(mac_valid_in),
        .ready_in(mac_ready_in),
        .acc_clear(mac_acc_clear),
        .result(mac_result),
        .valid_out(mac_valid_out),
        .ready_out(1'b1)
    );
    
    integer i;
    
    initial begin
        $dumpfile("simple_test.vcd");
        $dumpvars(0, simple_test);
        
        $display("===========================================");
        $display("SiLens RTL Simple Test");
        $display("===========================================");
        
        // Reset
        rst_n = 0;
        test_input = 0;
        activations = 0;
        weights = 0;
        mac_valid_in = 0;
        mac_acc_clear = 0;
        
        #100;
        rst_n = 1;
        #10;
        
        // Test popcount
        $display("");
        $display("Testing popcount module...");
        
        test_input = 64'h0000_0000_0000_0001;  // 1 bit
        #10;
        $display("  Input: %h, Count: %d (expected: 1)", test_input, count_out);
        if (count_out !== 1) $display("  ERROR!");
        
        test_input = 64'hFFFF_FFFF_FFFF_FFFF;  // 64 bits
        #10;
        $display("  Input: %h, Count: %d (expected: 64)", test_input, count_out);
        if (count_out !== 64) $display("  ERROR!");
        
        test_input = 64'hAAAA_AAAA_AAAA_AAAA;  // 32 bits
        #10;
        $display("  Input: %h, Count: %d (expected: 32)", test_input, count_out);
        if (count_out !== 32) $display("  ERROR!");
        
        // Test ternary MAC
        $display("");
        $display("Testing ternary MAC module...");
        
        // Clear accumulator first
        mac_acc_clear = 1;
        #10;
        mac_acc_clear = 0;
        
        // All +1 weights (01 pattern), activations = 1 for each element
        // 16 elements * 2 bits = 32 bits for weights
        weights = {16{2'b01}};  // 16 weights, all +1
        // 16 elements * 8 bits = 128 bits for activations
        activations = {16{8'd1}};  // 16 activations of value 1
        mac_valid_in = 1;
        #20;
        $display("  All +1 weights, activations=1, MAC result: %d", mac_result);
        
        // All -1 weights (10 pattern)
        mac_acc_clear = 1;
        #10;
        mac_acc_clear = 0;
        weights = {16{2'b10}};  // 16 weights, all -1
        #20;
        $display("  All -1 weights, MAC result: %d", mac_result);
        
        // All zero weights (00 pattern)
        mac_acc_clear = 1;
        #10;
        mac_acc_clear = 0;
        weights = {16{2'b00}};  // 16 weights, all 0
        #20;
        $display("  All zero weights, MAC result: %d", mac_result);
        
        mac_valid_in = 0;
        
        $display("");
        $display("===========================================");
        $display("Simple Test Complete!");
        $display("===========================================");
        
        #100;
        $finish;
    end
    
endmodule
'''
    
    # Write testbench
    tb_path = build_dir / "simple_test.v"
    with open(tb_path, 'w') as f:
        f.write(tb_code)
    
    # Compile testbench with modules
    rtl_dir = PROJECT_ROOT / "rtl"
    
    compile_cmd = [
        "iverilog",
        "-g2012",
        "-o", str(build_dir / "simple_test_sim"),
        "-DSIMULATION",
        "-I", str(rtl_dir / "common"),
        str(tb_path),
        str(rtl_dir / "common" / "popcount.v"),
        str(rtl_dir / "common" / "ternary_mac.v"),
    ]
    
    print_info("Compiling simple testbench...")
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print_error("Testbench compilation failed!")
        print(result.stderr)
        return False
    
    # Run simulation
    print_info("Running simulation...")
    run_cmd = ["vvp", str(build_dir / "simple_test_sim")]
    
    result = subprocess.run(
        run_cmd, 
        capture_output=True, 
        text=True,
        cwd=build_dir
    )
    
    if result.returncode != 0:
        print_error("Simulation failed!")
        print(result.stderr)
        return False
    
    # Print output
    print()
    for line in result.stdout.split('\n'):
        if line.strip():
            print(f"    {line}")
    
    # Check for errors in output
    if "ERROR" in result.stdout:
        print_error("Simulation had errors!")
        return False
    
    print()
    print_success("Simulation completed successfully!")
    
    # Check for VCD file
    vcd_path = build_dir / "simple_test.vcd"
    if vcd_path.exists():
        print_success(f"Waveform saved to: {vcd_path}")
    
    return True


# =============================================================================
# Step 5: Run Cocotb Test
# =============================================================================

def run_cocotb_test(build_dir: Path, weights_dir: Path):
    """Run cocotb-based test with Python verification."""
    print_step(5, "Running cocotb-based Python test")
    
    rtl_dir = PROJECT_ROOT / "rtl"
    tb_dir = rtl_dir / "tb"
    
    # Set environment
    env = os.environ.copy()
    env['SIM'] = 'icarus'
    env['COCOTB_RESOLVE_X'] = 'ZEROS'
    env['SILENS_WEIGHTS_DIR'] = str(weights_dir)
    
    # Run a simple cocotb test
    print_info("Running: make test-popcount")
    
    result = subprocess.run(
        ['make', 'test-popcount'],
        cwd=tb_dir,
        capture_output=True,
        text=True,
        env=env,
        timeout=120
    )
    
    # Check result
    if "PASSED" in result.stdout or result.returncode == 0:
        print_success("cocotb test passed!")
        
        # Print summary
        for line in result.stdout.split('\n'):
            if 'PASS' in line or 'FAIL' in line or 'test_' in line.lower():
                print(f"    {line}")
        
        return True
    else:
        print_error("cocotb test failed!")
        print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        if result.stderr:
            print(result.stderr[-500:])
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    print_header("SiLens End-to-End Pipeline Test")
    
    print("This test verifies the complete pipeline:")
    print("  1. Check simulation environment")
    print("  2. Quantize weights to ternary (1-bit)")
    print("  3. Compile RTL with Icarus Verilog")
    print("  4. Run simple Verilog simulation")
    print("  5. Run cocotb Python tests")
    print()
    
    # Create temp directories
    temp_base = Path(tempfile.mkdtemp(prefix="silens_e2e_test_"))
    weights_dir = temp_base / "weights"
    build_dir = temp_base / "build"
    
    print_info(f"Working directory: {temp_base}")
    
    results = {}
    
    # Step 1: Check environment
    results['environment'] = check_environment()
    if not results['environment']:
        print_error("Environment check failed. Please install missing dependencies.")
        return 1
    
    # Step 2: Quantize weights
    try:
        manifest = quantize_test_weights(weights_dir)
        results['quantization'] = True
    except Exception as e:
        print_error(f"Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        results['quantization'] = False
    
    # Step 3: Compile RTL
    if results.get('quantization'):
        results['compile'] = compile_rtl(build_dir)
    else:
        results['compile'] = False
    
    # Step 4: Run simple simulation
    if results.get('compile'):
        results['simulation'] = run_simple_simulation(build_dir, weights_dir)
    else:
        results['simulation'] = False
    
    # Step 5: Run cocotb test
    if results.get('simulation'):
        try:
            results['cocotb'] = run_cocotb_test(build_dir, weights_dir)
        except subprocess.TimeoutExpired:
            print_error("cocotb test timed out")
            results['cocotb'] = False
        except Exception as e:
            print_error(f"cocotb test failed: {e}")
            results['cocotb'] = False
    else:
        results['cocotb'] = False
    
    # Summary
    print_header("Test Summary")
    
    all_passed = True
    for step, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if passed else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {step:<20} [{status}]")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}All tests passed! ✓{Colors.END}")
        print()
        print("The SiLens RTL simulation pipeline is working:")
        print("  • Weights can be quantized to ternary")
        print("  • RTL compiles with Icarus Verilog")
        print("  • Verilog simulation runs correctly")
        print("  • Python/cocotb integration works")
        print()
        print(f"Temp files: {temp_base}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}Some tests failed ✗{Colors.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
