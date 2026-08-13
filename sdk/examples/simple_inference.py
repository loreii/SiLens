#!/usr/bin/env python3
"""
SiLens Simple Inference Example.

This example demonstrates basic usage of the SiLens SDK for
vision-language inference. It works in both simulation mode
(without hardware) and with real SiLens hardware.

Usage:
    # Basic usage (auto-detects hardware or uses simulation)
    python simple_inference.py image.jpg
    
    # With custom prompt
    python simple_inference.py image.jpg --prompt "What objects are in this image?"
    
    # Force simulation mode
    python simple_inference.py image.jpg --simulation
    
    # Benchmark mode
    python simple_inference.py image.jpg --benchmark --iterations 100
    
    # Streaming output
    python simple_inference.py image.jpg --stream
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from silens import (
    SiLensDevice,
    SimulatedDevice,
    InferenceEngine,
    load_image,
    Timer,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on an image using SiLens hardware"
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
        "--simulation", "-s",
        action="store_true",
        help="Force simulation mode (no hardware required)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output tokens as they are generated",
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run benchmark mode",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=10,
        help="Number of iterations for benchmark (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check input image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"SiLens Inference Example")
    print(f"========================")
    print(f"Image: {image_path}")
    print(f"Prompt: {args.prompt}")
    print()
    
    # Discover device
    print("Discovering devices...")
    mode = "simulation" if args.simulation else "auto"
    
    try:
        devices = SiLensDevice.discover(mode=mode)
        device = devices[0]
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Error discovering devices: {e}")
        sys.exit(1)
    
    # Create inference engine
    with device:
        engine = InferenceEngine(
            device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        if args.benchmark:
            run_benchmark(engine, image_path, args)
        elif args.stream:
            run_streaming(engine, image_path, args)
        else:
            run_single(engine, image_path, args)
        
        engine.close()


def run_single(engine: InferenceEngine, image_path: Path, args):
    """Run single inference."""
    print("Running inference...")
    print()
    
    result = engine.run(str(image_path), args.prompt)
    
    print("Output:")
    print("-" * 40)
    print(result.text)
    print("-" * 40)
    print()
    print(result.summary())



def run_streaming(engine: InferenceEngine, image_path: Path, args):
    """Run inference with streaming output."""
    print("Running inference (streaming)...")
    print()
    print("Output:")
    print("-" * 40)
    
    start_time = time.time()
    token_count = 0
    
    for token_text in engine.stream(str(image_path), args.prompt):
        print(token_text, end="", flush=True)
        token_count += 1
    
    elapsed = time.time() - start_time
    
    print()
    print("-" * 40)
    print()
    print(f"Generated {token_count} tokens in {elapsed*1000:.1f}ms")
    print(f"Speed: {token_count / elapsed:.1f} tokens/sec")


def run_benchmark(engine: InferenceEngine, image_path: Path, args):
    """Run benchmark mode."""
    print(f"Running benchmark ({args.iterations} iterations)...")
    print()
    
    times = []
    tokens_per_sec_list = []
    
    # Warmup
    print("Warmup run...")
    _ = engine.run(str(image_path), args.prompt)
    
    # Benchmark runs
    for i in range(args.iterations):
        result = engine.run(str(image_path), args.prompt)
        times.append(result.total_time_ms)
        tokens_per_sec_list.append(result.tokens_per_second)
        
        if args.verbose:
            print(f"  Run {i+1}: {result.total_time_ms:.1f}ms, "
                  f"{result.tokens_per_second:.1f} tok/s")
    
    # Calculate statistics
    import statistics
    
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    
    avg_tps = statistics.mean(tokens_per_sec_list)
    
    print()
    print("Benchmark Results:")
    print("-" * 40)
    print(f"Iterations: {args.iterations}")
    print(f"Latency:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"Throughput:")
    print(f"  Average: {avg_tps:.1f} tokens/sec")
    print(f"  Images:  {1000/avg_time:.1f} images/sec")


if __name__ == "__main__":
    main()
