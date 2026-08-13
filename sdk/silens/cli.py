#!/usr/bin/env python3
"""
SiLens Command-Line Interface.

Provides command-line tools for:
- Device information and diagnostics
- Running inference on images
- Performance benchmarking
- Model conversion utilities

Usage:
    silens info              # Show device information
    silens infer IMAGE       # Run inference on an image
    silens benchmark IMAGE   # Run performance benchmark
    silens convert MODEL     # Convert model for SiLens
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_version() -> str:
    """Get SDK version."""
    try:
        from silens import __version__
        return __version__
    except ImportError:
        return "unknown"


# =============================================================================
# Info Command
# =============================================================================

def info_cmd(args: argparse.Namespace) -> int:
    """Display device and SDK information."""
    from silens import SiLensDevice, __version__
    from silens.model import load_model_config
    
    print(f"\nSiLens SDK v{__version__}")
    print("=" * 50)
    
    # Discover devices
    print("\n📡 Discovering devices...")
    mode = "simulation" if args.simulation else "auto"
    
    try:
        devices = SiLensDevice.discover(mode=mode)
    except Exception as e:
        print(f"❌ Error discovering devices: {e}")
        return 1
    
    if not devices:
        print("⚠️  No devices found")
        return 1
    
    # Display device info
    print(f"\n🔌 Found {len(devices)} device(s):\n")
    
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
        
        try:
            with device:
                version = device.get_version()
                status = device.get_status()
                
                print(f"    Hardware Version: {version[0]}.{version[1]}.{version[2]}")
                print(f"    Status: {'Ready' if device.is_ready() else 'Busy'}")
                print(f"    Status Bits: 0x{int(status):04x}")
        except Exception as e:
            print(f"    ⚠️  Could not read device info: {e}")
        
        print()
    
    # Model info
    if args.verbose:
        print("📊 Model Configuration:")
        config = load_model_config()
        print(f"    Name: {config.name}")
        print(f"    Vision: {config.vision.name} ({config.vision.hidden_dim}d, {config.vision.num_layers} layers)")
        print(f"    Language: {config.language.name} ({config.language.hidden_dim}d, {config.language.num_layers} layers)")
        print(f"    Parameters: {config.total_parameters_str}")
        print(f"    Quantization: {config.weight_bits}-bit weights, {config.activation_bits}-bit activations")
        print()
    
    return 0


def setup_info_parser(subparsers: argparse._SubParsersAction) -> None:
    """Setup the info subcommand parser."""
    parser = subparsers.add_parser(
        "info",
        help="Display device and SDK information",
        description="Show information about connected SiLens devices and SDK configuration.",
    )
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Force simulation mode",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including model info",
    )
    parser.set_defaults(func=info_cmd)


# =============================================================================
# Infer Command
# =============================================================================

def infer_cmd(args: argparse.Namespace) -> int:
    """Run inference on an image."""
    from silens import SiLensDevice, InferenceEngine
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        return 1
    
    print(f"\n🖼️  Image: {image_path}")
    print(f"💬 Prompt: {args.prompt}")
    print()
    
    # Discover device
    mode = "simulation" if args.simulation else "auto"
    
    try:
        devices = SiLensDevice.discover(mode=mode)
        device = devices[0]
        print(f"🔌 Using device: {device}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Run inference
    try:
        with device:
            engine = InferenceEngine(
                device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            
            if args.stream:
                # Streaming output
                print("\n📝 Output:")
                print("-" * 40)
                
                start = time.time()
                token_count = 0
                
                for token in engine.stream(str(image_path), args.prompt):
                    print(token, end="", flush=True)
                    token_count += 1
                
                elapsed = time.time() - start
                print()
                print("-" * 40)
                print(f"\n⏱️  {token_count} tokens in {elapsed*1000:.1f}ms ({token_count/elapsed:.1f} tok/s)")
            else:
                # Single output
                print("⏳ Running inference...")
                result = engine.run(str(image_path), args.prompt)
                
                print("\n📝 Output:")
                print("-" * 40)
                print(result.text)
                print("-" * 40)
                
                if args.verbose:
                    print(f"\n{result.summary()}")
            
            engine.close()
    
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


def setup_infer_parser(subparsers: argparse._SubParsersAction) -> None:
    """Setup the infer subcommand parser."""
    parser = subparsers.add_parser(
        "infer",
        help="Run inference on an image",
        description="Run vision-language inference on an image using SiLens hardware.",
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Describe this image in detail.",
        help="Prompt for the model (default: 'Describe this image in detail.')",
    )
    parser.add_argument(
        "--max-tokens", "-m",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output tokens as generated",
    )
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Force simulation mode",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output",
    )
    parser.set_defaults(func=infer_cmd)


# =============================================================================
# Benchmark Command
# =============================================================================

def benchmark_cmd(args: argparse.Namespace) -> int:
    """Run performance benchmark."""
    from silens import SiLensDevice, InferenceEngine
    from silens.benchmark import Benchmark, generate_test_image
    
    # Get test image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Error: Image not found: {image_path}")
            return 1
        test_image = str(image_path)
        print(f"📷 Using image: {image_path}")
    else:
        print("📷 Using generated test image")
        test_image = generate_test_image()
    
    print(f"🔄 Iterations: {args.iterations} (warmup: {args.warmup})")
    print(f"🎯 Max tokens: {args.max_tokens}")
    print()
    
    # Discover device
    mode = "simulation" if args.simulation else "auto"
    
    try:
        devices = SiLensDevice.discover(mode=mode)
        device = devices[0]
        print(f"🔌 Using device: {device}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Run benchmark
    try:
        with device:
            engine = InferenceEngine(device, max_new_tokens=args.max_tokens)
            benchmark = Benchmark(engine, name=args.name or "silens-cli-benchmark")
            
            print("\n⏳ Running benchmark...")
            result = benchmark.run_latency(
                image=test_image,
                prompt=args.prompt,
                iterations=args.iterations,
                warmup=args.warmup,
                max_new_tokens=args.max_tokens,
            )
            
            # Show results
            result.print_summary()
            
            # Save results if requested
            if args.output:
                result.save(args.output)
                print(f"💾 Results saved to: {args.output}")
            
            engine.close()
    
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


def setup_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:
    """Setup the benchmark subcommand parser."""
    parser = subparsers.add_parser(
        "benchmark",
        help="Run performance benchmark",
        description="Run performance benchmarks on SiLens hardware.",
    )
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        help="Path to test image (uses generated image if not specified)",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Describe this image briefly.",
        help="Prompt for benchmark (default: 'Describe this image briefly.')",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--max-tokens", "-m",
        type=int,
        default=64,
        help="Maximum tokens per iteration (default: 64)",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Benchmark name for results",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Force simulation mode",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output",
    )
    parser.set_defaults(func=benchmark_cmd)


# =============================================================================
# Convert Command
# =============================================================================

def convert_cmd(args: argparse.Namespace) -> int:
    """Convert a model for SiLens hardware."""
    print(f"\n🔄 Model Conversion")
    print("=" * 50)
    print(f"📥 Input: {args.model}")
    print(f"📤 Output: {args.output or 'silens_model/'}")
    print()
    
    # Check if model path exists
    model_path = Path(args.model)
    if not model_path.exists():
        # Treat as HuggingFace model name
        print(f"🌐 Downloading model from HuggingFace: {args.model}")
    else:
        print(f"📂 Loading local model: {model_path}")
    
    # Placeholder for actual conversion
    # In a real implementation, this would call the model conversion pipeline
    print("\n⚠️  Model conversion is not yet fully implemented.")
    print("    See model/conversion/ for conversion scripts.")
    print()
    
    if args.analyze:
        print("📊 Model Analysis:")
        print("    This would show model architecture and layer info.")
    
    if args.quantize:
        print(f"\n🔢 Quantization: {args.bits}-bit")
        print("    This would quantize the model to ternary weights.")
    
    return 0


def setup_convert_parser(subparsers: argparse._SubParsersAction) -> None:
    """Setup the convert subcommand parser."""
    parser = subparsers.add_parser(
        "convert",
        help="Convert a model for SiLens hardware",
        description="Convert and optimize models for SiLens hardware acceleration.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model path or HuggingFace model name",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for converted model",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze model architecture",
    )
    parser.add_argument(
        "--quantize", "-q",
        action="store_true",
        help="Quantize model to ternary weights",
    )
    parser.add_argument(
        "--bits", "-b",
        type=int,
        default=2,
        choices=[1, 2, 4, 8],
        help="Quantization bits (default: 2 for ternary)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output",
    )
    parser.set_defaults(func=convert_cmd)


# =============================================================================
# Main Entry Point
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="silens",
        description="SiLens Vision-Language AI Accelerator CLI",
        epilog="For more information, visit: https://github.com/silens/silens-sdk",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"silens {get_version()}",
    )
    
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
    )
    
    # Setup subcommands
    setup_info_parser(subparsers)
    setup_infer_parser(subparsers)
    setup_benchmark_parser(subparsers)
    setup_convert_parser(subparsers)
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
