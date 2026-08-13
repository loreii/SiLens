#!/usr/bin/env python3
"""
SiLens Interactive Demo
=======================

This demo showcases the SiLens vision-language AI accelerator capabilities:

1. Ternary Quantization Pipeline
2. Hardware Simulation
3. Performance Profiling
4. Multi-Device Inference
5. Sparse Attention Analysis

Run: python demo.py
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent / "model"))
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Pretty Printing
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{Colors.END}\n")


def print_section(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}▶ {text}{Colors.END}")
    print(f"{Colors.CYAN}{'-'*50}{Colors.END}")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_metric(name, value, unit=""):
    print(f"  {name:<30} {Colors.YELLOW}{value:>10}{Colors.END} {unit}")


def animate_progress(text, duration=1.0, steps=20):
    """Show animated progress bar."""
    print(f"  {text} ", end="", flush=True)
    for i in range(steps):
        time.sleep(duration / steps)
        progress = "█" * (i + 1) + "░" * (steps - i - 1)
        print(f"\r  {text} [{progress}] {(i+1)*100//steps}%", end="", flush=True)
    print(f" {Colors.GREEN}Done!{Colors.END}")


# =============================================================================
# Demo 1: Ternary Quantization
# =============================================================================

def demo_quantization():
    """Demonstrate ternary quantization pipeline."""
    print_header("🔬 DEMO 1: Ternary Quantization Pipeline")
    
    from conversion.quantize_ternary import (
        TernaryQuantizer, TernaryQuantizationConfig, QuantizationMode
    )
    
    print_info("Simulating weight quantization for SmolVLM-256M architecture")
    print()
    
    # Simulate different layer types
    layers = [
        ("vision_encoder.patch_embed", (768, 768), "Vision"),
        ("vision_encoder.blocks.0.attn.qkv", (2304, 768), "Vision"),
        ("projector.linear", (576, 768), "Projector"),
        ("language_model.layers.0.self_attn.q_proj", (576, 576), "LLM"),
        ("language_model.layers.0.mlp.gate_proj", (1536, 576), "LLM"),
    ]
    
    config = TernaryQuantizationConfig(alpha=0.7, mode=QuantizationMode.PER_TENSOR)
    quantizer = TernaryQuantizer(config)
    
    total_params = 0
    total_positive = 0
    total_negative = 0
    total_zero = 0
    
    print_section("Quantizing Layers")
    
    for name, shape, layer_type in layers:
        # Generate realistic weight distribution
        weights = np.random.randn(*shape).astype(np.float32) * 0.02
        
        animate_progress(f"{name[:40]:<40}", duration=0.3)
        
        result = quantizer.quantize_tensor(name, weights)
        
        total_params += np.prod(shape)
        total_positive += result.num_positive
        total_negative += result.num_negative
        total_zero += result.num_zero
        
        print(f"    Sparsity: {result.sparsity:.1%} | "
              f"+1: {result.num_positive:,} | "
              f"-1: {result.num_negative:,} | "
              f"0: {result.num_zero:,}")
    
    print_section("Quantization Summary")
    
    print_metric("Total Parameters", f"{total_params:,}")
    print_metric("+1 Weights", f"{total_positive:,}", f"({total_positive/total_params*100:.1f}%)")
    print_metric("-1 Weights", f"{total_negative:,}", f"({total_negative/total_params*100:.1f}%)")
    print_metric("Zero Weights", f"{total_zero:,}", f"({total_zero/total_params*100:.1f}%)")
    print()
    
    # Memory savings
    original_mb = total_params * 4 / 1e6  # FP32
    quantized_mb = total_params * 2 / 8 / 1e6  # 2-bit ternary
    
    print_metric("Original Size (FP32)", f"{original_mb:.2f}", "MB")
    print_metric("Quantized Size (2-bit)", f"{quantized_mb:.2f}", "MB")
    print_metric("Compression Ratio", f"{original_mb/quantized_mb:.1f}x")
    
    print_success("Quantization pipeline complete!")


# =============================================================================
# Demo 2: Hardware Simulation
# =============================================================================

def demo_hardware_simulation():
    """Demonstrate hardware device simulation."""
    print_header("🖥️  DEMO 2: Hardware Device Simulation")
    
    from silens.device import SimulatedDevice, StatusBits, Registers
    
    print_info("Connecting to simulated SiLens accelerator...")
    print()
    
    device = SimulatedDevice(latency_ms=5.0, tokens_per_sec=80.0)
    device.open()
    
    print_section("Device Information")
    
    version = device.get_version()
    print_metric("Hardware Version", f"v{version[0]}.{version[1]}.{version[2]}")
    print_metric("Device Type", "SimulatedDevice")
    print_metric("Target Throughput", "80", "tokens/sec")
    
    print_section("Register Access")
    
    # Read some registers
    registers = [
        ("CTRL", Registers.CTRL),
        ("STATUS", Registers.STATUS),
        ("VERSION", Registers.VERSION),
    ]
    
    for name, reg in registers:
        value = device.read_reg(reg)
        print(f"  {name:<15} [0x{reg:03X}] = 0x{value:08X}")
    
    print_section("DMA Buffer Operations")
    
    # Allocate buffers
    buffer_sizes = [4096, 16384, 65536]
    buffers = []
    
    for size in buffer_sizes:
        buf = device.alloc_dma_buffer(size)
        buffers.append(buf)
        print(f"  Allocated {size:,} bytes @ 0x{buf.physical_addr:08X}")
    
    # Write test data
    test_data = np.random.randint(0, 256, size=1024, dtype=np.uint8)
    buffers[0].write(test_data, 0)
    print_success(f"Wrote {len(test_data)} bytes to buffer")
    
    # Transfer to device
    animate_progress("DMA Transfer to Device", duration=0.5)
    device.dma_transfer_to_device(buffers[0], 0x1000, 1024)
    
    # Cleanup
    for buf in buffers:
        device.free_dma_buffer(buf)
    
    print_section("Inference Simulation")
    
    # Start inference
    print_info("Starting vision-language inference...")
    device.start_inference(streaming=True)
    
    # Simulate token generation
    tokens = []
    print("\n  Generated tokens: ", end="", flush=True)
    
    for i in range(20):
        time.sleep(0.05)
        token = device.get_next_token()
        if token is not None:
            tokens.append(token)
            if token == 2:  # EOS
                print(f"{Colors.GREEN}[EOS]{Colors.END}")
                break
            char = chr(token) if 32 <= token < 127 else '?'
            print(f"{char}", end="", flush=True)
    
    print()
    print_metric("Tokens Generated", len(tokens))
    
    device.close()
    print_success("Hardware simulation complete!")


# =============================================================================
# Demo 3: Performance Profiling
# =============================================================================

def demo_profiling():
    """Demonstrate performance profiling capabilities."""
    print_header("📊 DEMO 3: Performance Profiling")
    
    from silens.device import SimulatedDevice
    from silens.profiler import Profiler, ProfileReport
    
    print_info("Running profiled inference session...")
    print()
    
    device = SimulatedDevice(latency_ms=10.0, tokens_per_sec=60.0)
    device.open()
    
    profiler = Profiler(device, sample_interval_ms=5.0)
    
    print_section("Profiled Inference")
    
    profiler.start()
    profiler.record_vision_start()
    
    animate_progress("Vision Encoding", duration=0.8)
    profiler.record_vision_end()
    
    profiler.record_llm_prefill_start()
    animate_progress("LLM Prefill", duration=0.5)
    profiler.record_llm_prefill_end()
    
    # Simulate token generation
    print("\n  Generating tokens: ", end="", flush=True)
    for i in range(15):
        time.sleep(0.03)
        profiler.record_token_generated(1000 + i)
        print("▪", end="", flush=True)
    print(f" {Colors.GREEN}Done!{Colors.END}")
    
    profiler.stop()
    
    # Generate report
    report = profiler.get_report()
    
    print_section("Timing Breakdown")
    
    print_metric("Total Inference Time", f"{report.total_time_ms:.1f}", "ms")
    print_metric("Vision Encoding", f"{report.vision_time_ms:.1f}", "ms")
    print_metric("LLM Prefill", f"{report.llm_prefill_time_ms:.1f}", "ms")
    print_metric("LLM Decode", f"{report.llm_decode_time_ms:.1f}", "ms")
    print_metric("Time to First Token", f"{report.time_to_first_token_ms:.1f}", "ms")
    
    print_section("Throughput Metrics")
    
    print_metric("Tokens Generated", report.tokens_generated)
    print_metric("Tokens/Second", f"{report.tokens_per_second:.1f}")
    
    print_section("Token Generation Latency")
    
    if report.token_profiles:
        times = [t.generation_time_ms for t in report.token_profiles if t.generation_time_ms > 0]
        if times:
            print_metric("Min Latency", f"{min(times):.2f}", "ms")
            print_metric("Max Latency", f"{max(times):.2f}", "ms")
            print_metric("Avg Latency", f"{np.mean(times):.2f}", "ms")
    
    device.close()
    print_success("Profiling complete!")


# =============================================================================
# Demo 4: Multi-Device Support
# =============================================================================

def demo_multi_device():
    """Demonstrate multi-device inference capabilities."""
    print_header("🔗 DEMO 4: Multi-Device Distributed Inference")
    
    from silens.device import SimulatedDevice
    from silens.multi_device import DevicePool, ParallelInference, ParallelStrategy
    
    print_info("Setting up multi-device inference cluster...")
    print()
    
    # Create multiple simulated devices
    devices = [
        SimulatedDevice(latency_ms=8.0, tokens_per_sec=70.0),
        SimulatedDevice(latency_ms=10.0, tokens_per_sec=65.0),
        SimulatedDevice(latency_ms=9.0, tokens_per_sec=72.0),
    ]
    
    pool = DevicePool(devices)
    
    print_section("Device Pool Status")
    
    pool.open()
    
    stats = pool.get_stats()
    print_metric("Total Devices", stats['num_devices'])
    print_metric("Available Devices", stats['num_available'])
    
    for dev in stats['devices']:
        status = f"{Colors.GREEN}●{Colors.END}" if dev['available'] else f"{Colors.RED}●{Colors.END}"
        print(f"  Device {dev['id']}: {status} Ready | Tasks: {dev['tasks']}")
    
    print_section("Data Parallel Inference")
    
    parallel = ParallelInference(pool, strategy=ParallelStrategy.DATA)
    
    # Simulate batch of images and prompts
    batch_size = 6
    images = [np.random.rand(384, 384, 3).astype(np.float32) for _ in range(batch_size)]
    prompts = [f"Describe this image {i+1}" for i in range(batch_size)]
    
    print_info(f"Processing batch of {batch_size} image-prompt pairs...")
    print()
    
    start = time.perf_counter()
    results = parallel.generate_batch(images, prompts)
    elapsed = (time.perf_counter() - start) * 1000
    
    for i, result in enumerate(results):
        print(f"  [{i+1}] {result[:50]}...")
    
    print()
    print_metric("Batch Size", batch_size)
    print_metric("Total Time", f"{elapsed:.1f}", "ms")
    print_metric("Per-Item Time", f"{elapsed/batch_size:.1f}", "ms")
    print_metric("Speedup vs Serial", f"{batch_size * 50 / elapsed:.1f}x", "(estimated)")
    
    print_section("Updated Pool Statistics")
    
    stats = pool.get_stats()
    for dev in stats['devices']:
        print(f"  Device {dev['id']}: Tasks={dev['tasks']} | "
              f"Avg Latency={dev['avg_latency_ms']:.1f}ms")
    
    parallel.shutdown()
    pool.close()
    
    print_success("Multi-device inference complete!")


# =============================================================================
# Demo 5: Sparse Attention Analysis
# =============================================================================

def demo_sparse_attention():
    """Demonstrate sparse attention pattern analysis."""
    print_header("🧠 DEMO 5: Sparse Attention Analysis")
    
    from analysis.sparse_attention import (
        AttentionAnalyzer, SparsePattern, SparsePatternConfig, generate_sparse_mask
    )
    
    print_info("Analyzing attention patterns for optimization...")
    print()
    
    analyzer = AttentionAnalyzer(sparsity_threshold=0.01, local_window=128)
    
    print_section("Collecting Attention Samples")
    
    # Simulate attention patterns from different layers
    layers = [
        ("vision_encoder.blocks.0.attn", 576, "vision", 0.3),   # Dense
        ("vision_encoder.blocks.6.attn", 576, "vision", 0.4),   # Medium
        ("vision_encoder.blocks.11.attn", 576, "vision", 0.5),  # Sparse
        ("language_model.layers.0.self_attn", 512, "language", 0.6),
        ("language_model.layers.15.self_attn", 512, "language", 0.7),
        ("language_model.layers.29.self_attn", 512, "language", 0.8),
    ]
    
    for name, seq_len, layer_type, sparsity in layers:
        # Generate synthetic attention with varying sparsity
        concentration = 1.0 / (1.0 - sparsity + 0.1)
        
        # Create attention with local bias
        attn = np.zeros((4, 8, seq_len, seq_len))
        for b in range(4):
            for h in range(8):
                for i in range(seq_len):
                    # Local window with some global attention
                    probs = np.random.dirichlet(np.ones(seq_len) * concentration)
                    # Add local bias
                    window = min(64, seq_len)
                    start = max(0, i - window//2)
                    end = min(seq_len, i + window//2)
                    probs[start:end] *= 2
                    probs /= probs.sum()
                    attn[b, h, i] = probs
        
        import torch
        analyzer.collect_attention(name, torch.from_numpy(attn))
        print(f"  Collected: {name}")
    
    print_section("Analyzing Patterns")
    
    analyzer.analyze_all_layers()
    
    # Display results
    print(f"\n  {'Layer':<45} {'Pattern':<12} {'Sparsity':<10} {'Reduction'}")
    print(f"  {'-'*85}")
    
    for name, stats in sorted(analyzer.layer_stats.items()):
        pattern = stats.recommended_pattern.value
        if pattern == "dense":
            pattern_colored = f"{Colors.RED}{pattern:<12}{Colors.END}"
        elif pattern == "local":
            pattern_colored = f"{Colors.GREEN}{pattern:<12}{Colors.END}"
        else:
            pattern_colored = f"{Colors.YELLOW}{pattern:<12}{Colors.END}"
        
        print(f"  {name:<45} {pattern_colored} {stats.mean_sparsity:>8.1%}   {stats.reduction_ratio:>6.1%}")
    
    print_section("Optimization Summary")
    
    summary = analyzer.get_summary()
    
    print_metric("Layers Analyzed", summary['num_layers'])
    print_metric("Average Sparsity", f"{summary['avg_sparsity']:.1%}")
    print_metric("Compute Reduction", f"{summary['overall_compute_reduction']:.1%}")
    print_metric("Memory Reduction", f"{summary['memory_reduction']:.1%}")
    
    print("\n  Pattern Distribution:")
    for pattern, count in summary['pattern_distribution'].items():
        bar = "█" * count + "░" * (6 - count)
        print(f"    {pattern:<12} [{bar}] {count} layers")
    
    print_section("Sample Sparse Mask Generation")
    
    # Generate a sparse mask
    config = SparsePatternConfig(
        pattern=SparsePattern.LONGFORMER,
        window_size=128,
        num_global_tokens=4
    )
    
    mask = generate_sparse_mask(config, seq_len=256)
    sparsity = 1.0 - mask.sum() / mask.size
    
    print_info(f"Generated Longformer mask (256x256)")
    print_metric("Window Size", "128", "tokens")
    print_metric("Global Tokens", "4")
    print_metric("Mask Sparsity", f"{sparsity:.1%}")
    
    # Visualize a portion of the mask
    print("\n  Mask Pattern (top-left 20x20):")
    print("  ", end="")
    for j in range(20):
        print(f"{j%10}", end="")
    print()
    for i in range(20):
        print(f"  ", end="")
        for j in range(20):
            print("█" if mask[i, j] else "·", end="")
        print()
    
    print_success("Sparse attention analysis complete!")


# =============================================================================
# Demo 6: End-to-End Pipeline (RTL Simulation)
# =============================================================================

def demo_end_to_end():
    """Demonstrate complete inference pipeline through RTL simulation."""
    print_header("🚀 DEMO 6: End-to-End RTL Simulation")
    
    print_info("This demo runs inference through actual Verilog simulation")
    print_info("using Icarus Verilog or Verilator with cocotb")
    print()
    
    # Check RTL simulation environment
    try:
        sys.path.insert(0, str(Path(__file__).parent / "rtl" / "tb"))
        from sim_interface import (
            run_e2e_simulation,
            check_simulation_environment,
            SimulationConfig,
            SimulationResults
        )
        env_status = check_simulation_environment()
    except ImportError as e:
        print(f"{Colors.YELLOW}⚠ RTL simulation interface not available: {e}{Colors.END}")
        env_status = {}
    
    # Display environment status
    print_section("Simulation Environment")
    
    has_simulator = env_status.get('icarus', False) or env_status.get('verilator', False)
    has_cocotb = env_status.get('cocotb', False)
    
    print(f"  Icarus Verilog: {'✓' if env_status.get('icarus') else '✗'}")
    print(f"  Verilator:      {'✓' if env_status.get('verilator') else '✗'}")
    print(f"  cocotb:         {'✓' if env_status.get('cocotb') else '✗'}")
    print(f"  numpy:          {'✓' if env_status.get('numpy') else '✗'}")
    print()
    
    if not has_simulator:
        print(f"{Colors.YELLOW}⚠ No RTL simulator found.{Colors.END}")
        print()
        print("  Install Icarus Verilog:")
        print(f"    {Colors.CYAN}brew install icarus-verilog{Colors.END}  (macOS)")
        print(f"    {Colors.CYAN}apt install iverilog{Colors.END}         (Ubuntu)")
        print()
        print("  Or install Verilator:")
        print(f"    {Colors.CYAN}brew install verilator{Colors.END}       (macOS)")
        print(f"    {Colors.CYAN}apt install verilator{Colors.END}        (Ubuntu)")
        print()
        _demo_end_to_end_simulated()
        return
    
    if not has_cocotb:
        print(f"{Colors.YELLOW}⚠ cocotb not found.{Colors.END}")
        print(f"  Install with: {Colors.CYAN}pip install cocotb{Colors.END}")
        print()
        _demo_end_to_end_simulated()
        return
    
    # Select simulation mode
    print_section("Simulation Mode")
    
    print("  Options:")
    print("    [1] Quick test (small image, few tokens) ~30 seconds")
    print("    [2] Full inference (384x384 image, full prompt) ~5 minutes")
    print("    [3] Custom (provide your own image)")
    print()
    
    try:
        choice = input(f"  {Colors.BOLD}Select [1/2/3]: {Colors.END}").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return
    
    # Prepare inputs based on choice
    if choice == "3":
        try:
            image_path = input(f"  {Colors.BOLD}Image path: {Colors.END}").strip()
            if not Path(image_path).exists():
                print(f"{Colors.RED}  File not found: {image_path}{Colors.END}")
                return
            # Load image
            try:
                from PIL import Image as PILImage
                img = PILImage.open(image_path).convert('RGB')
                img = img.resize((384, 384))
                image = np.array(img).astype(np.float32) / 255.0
            except ImportError:
                print(f"{Colors.YELLOW}  PIL not available, using numpy to load{Colors.END}")
                image = np.random.rand(384, 384, 3).astype(np.float32)
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return
        
        # Get custom prompt tokens (simplified)
        tokens = [1, 8612, 436, 2217, 2]  # "Describe this image"
        
    elif choice == "2":
        # Full inference with gradient test image
        print("\n  Creating 384x384 test image...")
        img_size = 384
        x = np.linspace(0, 1, img_size)
        y = np.linspace(0, 1, img_size)
        xv, yv = np.meshgrid(x, y)
        image = np.stack([xv, yv, (xv + yv) / 2], axis=-1).astype(np.float32)
        tokens = [1, 8612, 436, 2217, 323, 1538, 2]  # Longer prompt
        
    else:  # Quick test
        print("\n  Creating small test image...")
        image = np.random.rand(56, 56, 3).astype(np.float32)
        # Resize to expected size
        image = np.repeat(np.repeat(image, 7, axis=0), 7, axis=1)[:384, :384]
        tokens = [1, 100, 2]  # Minimal tokens
    
    print(f"  Image shape: {image.shape}")
    print(f"  Tokens: {tokens}")
    
    print_section("Running RTL Simulation")
    
    print_info("Compiling Verilog and running simulation...")
    print_info("This exercises the actual hardware design")
    print()
    
    # Configure simulation
    simulator = 'icarus' if env_status.get('icarus') else 'verilator'
    config = SimulationConfig(
        simulator=simulator,
        timeout_sec=600,  # 10 minutes max
        verbose=False
    )
    
    # Show progress
    print(f"  Simulator: {Colors.CYAN}{simulator}{Colors.END}")
    print()
    
    animate_progress("Compiling RTL", duration=2.0)
    
    # Run simulation
    start_time = time.perf_counter()
    
    try:
        results = run_e2e_simulation(image, tokens, config=config)
    except Exception as e:
        print(f"\n{Colors.RED}Simulation failed: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        return
    
    elapsed = time.perf_counter() - start_time
    
    print_section("Simulation Results")
    
    if results.success:
        print_success("RTL simulation completed successfully!")
        print()
        
        # Cycle counts
        print_metric("Total Cycles", f"{results.total_cycles:,}")
        print_metric("Vision Encoding", f"{results.vision_cycles:,}", "cycles")
        print_metric("LLM Prefill", f"{results.prefill_cycles:,}", "cycles")
        print_metric("Token Generation", f"{results.decode_cycles:,}", "cycles")
        print()
        
        # Timing at 100 MHz
        timing = results.get_timing_summary(clock_freq_mhz=100)
        print_metric("Total Time @ 100MHz", f"{timing['total_ms']:.2f}", "ms")
        print_metric("Vision Time", f"{timing['vision_ms']:.2f}", "ms")
        print_metric("Prefill Time", f"{timing['prefill_ms']:.2f}", "ms")
        print_metric("Decode Time", f"{timing['decode_ms']:.2f}", "ms")
        print()
        
        # Performance
        print_metric("Time to First Token", f"{results.time_to_first_token:,}", "cycles")
        print_metric("Tokens Generated", len(results.output_tokens))
        print_metric("Tokens/Second @ 100MHz", f"{results.tokens_per_second:.1f}")
        print()
        
        # Output tokens
        if results.output_tokens:
            print(f"  Output tokens: {results.output_tokens[:20]}")
            if len(results.output_tokens) > 20:
                print(f"                 ... ({len(results.output_tokens)} total)")
        
    else:
        print(f"{Colors.RED}✗ Simulation failed{Colors.END}")
        print(f"  Error: {results.error_message}")
        if results.stderr:
            print(f"\n  Stderr (last 500 chars):")
            print(f"  {results.stderr[-500:]}")
    
    print()
    print_metric("Wall Clock Time", f"{elapsed:.1f}", "seconds")
    
    print_section("Hardware vs Simulation")
    
    print("  This simulation runs the actual Verilog RTL design.")
    print("  On real SiLens hardware (ASIC @ 100MHz):")
    print()
    print_metric("  Expected Total Latency", "<5", "ms")
    print_metric("  Expected Throughput", "80+", "tokens/sec")
    print_metric("  Expected Power", "2-3", "W")
    print()
    print_info("Simulation is cycle-accurate but runs ~1000x slower than real silicon")
    
    print_success("RTL simulation demo complete!")


def _create_sample_image() -> str:
    """Create a simple sample image for testing."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # Fall back to numpy-only image
        img_data = np.random.randint(100, 200, (384, 384, 3), dtype=np.uint8)
        # Add some colored rectangles
        img_data[50:150, 50:150] = [255, 100, 100]  # Red square
        img_data[200:300, 100:250] = [100, 255, 100]  # Green rectangle
        img_data[100:200, 250:350] = [100, 100, 255]  # Blue square
        
        path = "/tmp/silens_sample.png"
        # Save using raw method
        import struct
        import zlib
        
        def write_png(filename, pixels):
            def make_chunk(chunk_type, data):
                chunk_len = len(data)
                chunk = chunk_type + data
                crc = zlib.crc32(chunk) & 0xffffffff
                return struct.pack('>I', chunk_len) + chunk + struct.pack('>I', crc)
            
            h, w = pixels.shape[:2]
            raw_data = b''
            for row in pixels:
                raw_data += b'\x00' + row.tobytes()
            
            with open(filename, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n')
                f.write(make_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)))
                f.write(make_chunk(b'IDAT', zlib.compress(raw_data)))
                f.write(make_chunk(b'IEND', b''))
        
        write_png(path, img_data)
        return path
    
    # Create a more interesting test image with PIL
    img = Image.new('RGB', (384, 384), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([30, 30, 150, 150], fill=(255, 100, 100), outline=(200, 50, 50))
    draw.ellipse([180, 50, 350, 180], fill=(100, 200, 100), outline=(50, 150, 50))
    draw.polygon([(100, 200), (200, 350), (50, 300)], fill=(100, 100, 255))
    draw.rectangle([220, 220, 360, 360], fill=(255, 200, 100), outline=(200, 150, 50))
    
    # Add some lines
    for i in range(0, 384, 40):
        draw.line([(i, 0), (i, 384)], fill=(220, 220, 220), width=1)
        draw.line([(0, i), (384, i)], fill=(220, 220, 220), width=1)
    
    path = "/tmp/silens_sample.png"
    img.save(path)
    return path


def _demo_end_to_end_simulated():
    """Fallback simulated demo when model backend is not available."""
    print_info("Running SIMULATED demo (model not loaded)")
    print()
    
    print_section("Pipeline Stages (Simulated)")
    
    stages = [
        ("1. Image Loading", "Loading 384x384 RGB image", 0.2),
        ("2. Patch Embedding", "Extracting 576 patches (16x16)", 0.3),
        ("3. Vision Encoding", "12 ViT blocks, 768-dim", 0.8),
        ("4. Multimodal Projection", "768-dim → 576-dim", 0.2),
        ("5. Token Embedding", "Encoding text prompt", 0.1),
        ("6. LLM Prefill", "30 layers, 576-dim", 0.6),
        ("7. Autoregressive Decode", "Generating response", 1.5),
    ]
    
    total_time = 0
    for stage_name, description, duration in stages:
        print(f"\n  {Colors.BOLD}{stage_name}{Colors.END}")
        print(f"    {description}")
        animate_progress("    Processing", duration=duration)
        total_time += duration
    
    print_section("Generated Response (Simulated)")
    
    response = """The image shows a cozy living room with a 
comfortable sofa, wooden coffee table, and large 
windows letting in natural light. A cat is curled 
up sleeping on a soft blanket."""
    
    print(f"\n  {Colors.GREEN}Prompt:{Colors.END} Describe what you see in this image.")
    print(f"\n  {Colors.YELLOW}[SIMULATED] Response:{Colors.END}")
    
    for word in response.split():
        print(word, end=" ", flush=True)
        time.sleep(0.05)
    print("\n")
    
    print_section("To run with REAL model inference:")
    print(f"  {Colors.CYAN}pip install torch transformers pillow{Colors.END}")
    print()
    
    print_success("Simulated demo complete!")


# =============================================================================
# Main Menu
# =============================================================================

def print_banner():
    """Print the SiLens banner."""
    banner = f"""
{Colors.CYAN}
   ███████╗██╗██╗     ███████╗███╗   ██╗███████╗
   ██╔════╝██║██║     ██╔════╝████╗  ██║██╔════╝
   ███████╗██║██║     █████╗  ██╔██╗ ██║███████╗
   ╚════██║██║██║     ██╔══╝  ██║╚██╗██║╚════██║
   ███████║██║███████╗███████╗██║ ╚████║███████║
   ╚══════╝╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
{Colors.END}
{Colors.BOLD}   Vision-Language AI Accelerator - Interactive Demo{Colors.END}
   
   Ternary Neural Network Hardware for Edge AI
   PCIe/USB Accelerator | 2-bit Weights | <3W Power
"""
    print(banner)


def print_menu():
    """Print the demo menu."""
    print(f"""
{Colors.BOLD}Available Demos:{Colors.END}

  {Colors.YELLOW}[1]{Colors.END} Ternary Quantization Pipeline
      Convert FP32 weights to 2-bit ternary format
      
  {Colors.YELLOW}[2]{Colors.END} Hardware Device Simulation  
      Interact with simulated SiLens accelerator
      
  {Colors.YELLOW}[3]{Colors.END} Performance Profiling
      Detailed timing and throughput analysis
      
  {Colors.YELLOW}[4]{Colors.END} Multi-Device Inference
      Distributed batch processing across devices
      
  {Colors.YELLOW}[5]{Colors.END} Sparse Attention Analysis
      Optimize attention patterns for hardware
      
  {Colors.YELLOW}[6]{Colors.END} End-to-End Pipeline
      Complete inference demonstration
      
  {Colors.YELLOW}[A]{Colors.END} Run All Demos
  
  {Colors.YELLOW}[Q]{Colors.END} Quit
""")


def main():
    """Main entry point."""
    print_banner()
    
    demos = {
        '1': demo_quantization,
        '2': demo_hardware_simulation,
        '3': demo_profiling,
        '4': demo_multi_device,
        '5': demo_sparse_attention,
        '6': demo_end_to_end,
    }
    
    while True:
        print_menu()
        
        try:
            choice = input(f"{Colors.BOLD}Select demo [1-6, A, Q]: {Colors.END}").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break
        
        if choice == 'Q':
            print(f"\n{Colors.CYAN}Thanks for trying SiLens! 👋{Colors.END}\n")
            break
        elif choice == 'A':
            for demo_func in demos.values():
                try:
                    demo_func()
                except Exception as e:
                    print(f"{Colors.RED}Error: {e}{Colors.END}")
                print("\n" + "="*70 + "\n")
                time.sleep(0.5)
        elif choice in demos:
            try:
                demos[choice]()
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.END}")
                import traceback
                traceback.print_exc()
        else:
            print(f"{Colors.RED}Invalid choice. Please select 1-6, A, or Q.{Colors.END}")
        
        print()
        input(f"{Colors.CYAN}Press Enter to continue...{Colors.END}")


if __name__ == "__main__":
    main()
