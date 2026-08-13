#!/usr/bin/env python3
"""
SiLens Batch Inference Example.

Demonstrates how to process multiple images efficiently using the SiLens SDK.
Includes progress tracking and result aggregation.

Usage:
    python batch_inference.py images/ --output results.json
    python batch_inference.py images/ --prompt "What objects are in this image?"
    python batch_inference.py images/ --max-images 100 --simulation
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from silens import SiLensDevice, InferenceEngine, Timer


def find_images(directory: Path, extensions: set = None) -> List[Path]:
    """Find all images in a directory."""
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    
    images = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            images.append(path)
    
    return images


def process_batch(
    engine: InferenceEngine,
    images: List[Path],
    prompt: str,
    max_tokens: int = 128,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Process a batch of images and return results."""
    results = []
    timer = Timer()
    
    total = len(images)
    timer.start("total")
    
    for i, image_path in enumerate(images):
        timer.start("inference")
        
        try:
            result = engine.run(str(image_path), prompt, max_new_tokens=max_tokens)
            
            results.append({
                "image": str(image_path),
                "text": result.text,
                "num_tokens": result.num_tokens,
                "time_ms": result.total_time_ms,
                "tokens_per_sec": result.tokens_per_second,
                "success": True,
            })
            
            if verbose:
                print(f"[{i+1}/{total}] {image_path.name}: {result.text[:50]}...")
            
        except Exception as e:
            results.append({
                "image": str(image_path),
                "error": str(e),
                "success": False,
            })
            if verbose:
                print(f"[{i+1}/{total}] {image_path.name}: ERROR - {e}")
        
        timer.stop("inference")
        
        # Progress indicator
        if not verbose and (i + 1) % 10 == 0:
            elapsed = timer.get("inference")
            rate = (i + 1) / (elapsed / 1000) if elapsed > 0 else 0
            print(f"Progress: {i+1}/{total} ({rate:.1f} img/s)")
    
    timer.stop("total")
    
    return results


def print_summary(results: List[Dict[str, Any]], elapsed_sec: float) -> None:
    """Print summary statistics."""
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    print(f"Total images: {len(results)}")
    print(f"Successful:   {len(successful)}")
    print(f"Failed:       {len(failed)}")
    print(f"Total time:   {elapsed_sec:.2f}s")
    print(f"Throughput:   {len(results) / elapsed_sec:.2f} img/s")
    
    if successful:
        times = [r["time_ms"] for r in successful]
        tokens = [r["num_tokens"] for r in successful]
        tps = [r["tokens_per_sec"] for r in successful]
        
        print(f"\nLatency (per image):")
        print(f"  Average: {sum(times)/len(times):.1f} ms")
        print(f"  Min:     {min(times):.1f} ms")
        print(f"  Max:     {max(times):.1f} ms")
        
        print(f"\nTokens generated:")
        print(f"  Total:   {sum(tokens)}")
        print(f"  Average: {sum(tokens)/len(tokens):.1f} per image")
        
        print(f"\nGeneration speed:")
        print(f"  Average: {sum(tps)/len(tps):.1f} tok/s")
    
    if failed:
        print(f"\nFailed images:")
        for r in failed[:5]:  # Show first 5
            print(f"  - {r['image']}: {r.get('error', 'Unknown error')}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process multiple images with SiLens"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input directory containing images",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Describe this image briefly.",
        help="Prompt for all images (default: 'Describe this image briefly.')",
    )
    parser.add_argument(
        "--max-images", "-n",
        type=int,
        default=None,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=128,
        help="Maximum tokens per image (default: 128)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results",
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
    
    args = parser.parse_args()
    
    # Find images
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    images = find_images(input_dir)
    if not images:
        print(f"Error: No images found in {input_dir}")
        sys.exit(1)
    
    if args.max_images:
        images = images[:args.max_images]
    
    print(f"Found {len(images)} images in {input_dir}")
    print(f"Prompt: {args.prompt}")
    print()
    
    # Discover device
    mode = "simulation" if args.simulation else "auto"
    devices = SiLensDevice.discover(mode=mode)
    device = devices[0]
    print(f"Using device: {device}")
    print()
    
    # Process images
    start_time = time.time()
    
    with device:
        engine = InferenceEngine(device, max_new_tokens=args.max_tokens)
        results = process_batch(engine, images, args.prompt, args.max_tokens, args.verbose)
        engine.close()
    
    elapsed = time.time() - start_time
    
    # Print summary
    print_summary(results, elapsed)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "prompt": args.prompt,
            "total_images": len(results),
            "total_time_sec": elapsed,
            "results": results,
        }
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
