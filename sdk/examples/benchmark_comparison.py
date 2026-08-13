#!/usr/bin/env python3
"""
SiLens Benchmark Comparison Example.

Compares SiLens hardware performance against CPU and (optional) GPU baselines.
Demonstrates the performance advantages of dedicated hardware acceleration.

Usage:
    python benchmark_comparison.py
    python benchmark_comparison.py --image test.jpg --iterations 50
    python benchmark_comparison.py --with-gpu --output results.json

Requirements for baseline comparisons:
    pip install transformers torch  # For CPU/GPU baseline
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from silens import SiLensDevice, InferenceEngine
from silens.benchmark import (
    Benchmark,
    ComparisonBenchmark,
    BenchmarkResult,
    generate_test_image,
)


def create_cpu_baseline() -> Optional[Callable]:
    """
    Create CPU baseline inference function.
    
    Returns None if transformers/torch not available.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from PIL import Image
        
        print("📥 Loading CPU baseline model...")
        
        # Use a small model for baseline comparison
        model_name = "HuggingFaceTB/SmolLM2-135M"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        model.eval()
        
        print("   CPU baseline ready\n")
        
        def cpu_inference(image, prompt: str) -> Dict[str, Any]:
            """Run inference on CPU."""
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )
            elapsed_ms = (time.time() - start) * 1000
            
            # Decode
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            
            return {
                "text": text,
                "num_tokens": num_tokens,
                "time_ms": elapsed_ms,
            }
        
        return cpu_inference
        
    except ImportError as e:
        print(f"⚠️  CPU baseline not available: {e}")
        print("   Install with: pip install transformers torch\n")
        return None


def create_gpu_baseline() -> Optional[Callable]:
    """
    Create GPU baseline inference function.
    
    Returns None if CUDA not available.
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping GPU baseline\n")
            return None
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("📥 Loading GPU baseline model...")
        
        model_name = "HuggingFaceTB/SmolLM2-135M"
        device = "cuda"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        model.eval()
        
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   GPU baseline ready ({gpu_name})\n")
        
        def gpu_inference(image, prompt: str) -> Dict[str, Any]:
            """Run inference on GPU."""
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Warmup
            torch.cuda.synchronize()
            
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            elapsed_ms = (time.time() - start) * 1000
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            
            return {
                "text": text,
                "num_tokens": num_tokens,
                "time_ms": elapsed_ms,
            }
        
        return gpu_inference
        
    except ImportError as e:
        print(f"⚠️  GPU baseline not available: {e}")
        return None


def run_comparison(
    engine: InferenceEngine,
    image,
    iterations: int = 50,
    warmup: int = 5,
    with_cpu: bool = True,
    with_gpu: bool = False,
) -> Dict[str, BenchmarkResult]:
    """Run benchmark comparison."""
    
    comparison = ComparisonBenchmark(engine)
    
    # Add baselines
    if with_cpu:
        cpu_fn = create_cpu_baseline()
        if cpu_fn:
            comparison.add_baseline("cpu", cpu_fn)
    
    if with_gpu:
        gpu_fn = create_gpu_baseline()
        if gpu_fn:
            comparison.add_baseline("gpu", gpu_fn)
    
    # Run comparison
    print(f"🏃 Running benchmark ({iterations} iterations, {warmup} warmup)...\n")
    
    results = comparison.run(
        image=image,
        prompt="Describe this image briefly.",
        iterations=iterations,
        warmup=warmup,
    )
    
    return results


def print_detailed_results(results: Dict[str, BenchmarkResult]) -> None:
    """Print detailed benchmark results."""
    
    print("\n" + "="*80)
    print("📊 Detailed Benchmark Results")
    print("="*80)
    
    silens_result = results.get("silens")
    if not silens_result:
        print("No SiLens results available")
        return
    
    # Calculate speedups
    base_latency = silens_result.total_latency.mean if silens_result.total_latency else 1.0
    
    for name, result in sorted(results.items()):
        print(f"\n{'─'*40}")
        print(f"📌 {name.upper()}")
        print(f"{'─'*40}")
        
        if result.total_latency:
            lat = result.total_latency
            print(f"  Latency:")
            print(f"    Mean:    {lat.mean:>8.2f} ms")
            print(f"    Std Dev: {lat.std:>8.2f} ms")
            print(f"    Min:     {lat.min:>8.2f} ms")
            print(f"    Max:     {lat.max:>8.2f} ms")
            print(f"    P50:     {lat.p50:>8.2f} ms")
            print(f"    P90:     {lat.p90:>8.2f} ms")
            print(f"    P99:     {lat.p99:>8.2f} ms")
            
            if name != "silens":
                speedup = lat.mean / base_latency
                print(f"    vs SiLens: {speedup:.2f}x slower")
        
        if result.tokens_per_second:
            tps = result.tokens_per_second
            print(f"  Throughput:")
            print(f"    Mean:    {tps.mean:>8.1f} tok/s")
        
        print(f"  Samples: {result.iterations}")
    
    print("\n" + "="*80)


def print_summary_table(results: Dict[str, BenchmarkResult]) -> None:
    """Print a summary comparison table."""
    
    silens = results.get("silens")
    if not silens or not silens.total_latency:
        return
    
    base_latency = silens.total_latency.mean
    base_tps = silens.tokens_per_second.mean if silens.tokens_per_second else 0
    
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║                    Performance Comparison                      ║")
    print("╠═══════════════╦════════════════╦════════════╦═════════════════╣")
    print("║    Device     ║  Latency (ms)  ║  Tok/sec   ║    Speedup      ║")
    print("╠═══════════════╬════════════════╬════════════╬═════════════════╣")
    
    for name in ["silens", "cpu", "gpu"]:
        if name not in results:
            continue
        
        result = results[name]
        lat = result.total_latency.mean if result.total_latency else 0
        tps = result.tokens_per_second.mean if result.tokens_per_second else 0
        
        if name == "silens":
            speedup = "1.0x (baseline)"
        else:
            speedup = f"{lat / base_latency:.1f}x slower"
        
        icon = {"silens": "⚡", "cpu": "💻", "gpu": "🎮"}.get(name, "")
        print(f"║ {icon} {name:<11} ║ {lat:>14.2f} ║ {tps:>10.1f} ║ {speedup:<15} ║")
    
    print("╚═══════════════╩════════════════╩════════════╩═════════════════╝")
    
    # Calculate advantages
    cpu_result = results.get("cpu")
    if cpu_result and cpu_result.total_latency:
        cpu_speedup = cpu_result.total_latency.mean / base_latency
        print(f"\n💡 SiLens is {cpu_speedup:.1f}x faster than CPU")
    
    gpu_result = results.get("gpu")
    if gpu_result and gpu_result.total_latency:
        gpu_speedup = gpu_result.total_latency.mean / base_latency
        print(f"💡 SiLens is {gpu_speedup:.1f}x faster than GPU")
    
    print("\n🔋 SiLens advantage: Dedicated silicon for VLM inference")
    print("   - Lower latency")
    print("   - Lower power consumption")
    print("   - No thermal throttling")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare SiLens performance against CPU/GPU baselines"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Test image (uses generated image if not specified)",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=50,
        help="Benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=5,
        help="Warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--no-cpu",
        action="store_true",
        help="Skip CPU baseline",
    )
    parser.add_argument(
        "--with-gpu",
        action="store_true",
        help="Include GPU baseline (requires CUDA)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Force simulation mode for SiLens",
    )
    
    args = parser.parse_args()
    
    # Banner
    print("""
╔═══════════════════════════════════════════════════════════╗
║           SiLens Performance Comparison                   ║
║        Hardware Accelerator vs CPU/GPU Baseline           ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    # Get test image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)
        test_image = str(image_path)
        print(f"📷 Test image: {image_path}")
    else:
        print("📷 Using generated test image (384x384)")
        test_image = generate_test_image()
    
    print()
    
    # Discover SiLens device
    mode = "simulation" if args.simulation else "auto"
    devices = SiLensDevice.discover(mode=mode)
    device = devices[0]
    print(f"⚡ SiLens device: {device}\n")
    
    # Run comparison
    with device:
        engine = InferenceEngine(device)
        
        results = run_comparison(
            engine,
            test_image,
            iterations=args.iterations,
            warmup=args.warmup,
            with_cpu=not args.no_cpu,
            with_gpu=args.with_gpu,
        )
        
        engine.close()
    
    # Print results
    print_summary_table(results)
    print_detailed_results(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            name: result.to_dict() for name, result in results.items()
        }
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
