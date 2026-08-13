"""
SiLens Benchmark Module.

Performance benchmarking utilities for SiLens hardware including:
- Latency measurement (vision encoder, token generation, end-to-end)
- Throughput testing (images/sec, tokens/sec)
- Memory bandwidth analysis
- Comparison with CPU/GPU baselines
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from silens.device import SiLensDevice
    from silens.inference import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class LatencyResult:
    """Results from a single latency measurement."""
    total_ms: float
    vision_ms: float
    generation_ms: float
    preprocessing_ms: float
    tokens_generated: int
    
    @property
    def tokens_per_second(self) -> float:
        """Token generation rate."""
        if self.generation_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.generation_ms / 1000.0)


@dataclass
class BenchmarkStats:
    """Statistical summary of benchmark results."""
    metric_name: str
    unit: str
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p99: float
    samples: int
    
    def __str__(self) -> str:
        return (
            f"{self.metric_name}: {self.mean:.2f} ± {self.std:.2f} {self.unit} "
            f"(p50={self.p50:.2f}, p90={self.p90:.2f}, p99={self.p99:.2f})"
        )
    
    @classmethod
    def from_values(
        cls,
        values: List[float],
        metric_name: str,
        unit: str,
    ) -> "BenchmarkStats":
        """Create stats from a list of values."""
        if not values:
            return cls(
                metric_name=metric_name,
                unit=unit,
                mean=0, std=0, min=0, max=0,
                p50=0, p90=0, p99=0, samples=0,
            )
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        return cls(
            metric_name=metric_name,
            unit=unit,
            mean=statistics.mean(values),
            std=statistics.stdev(values) if n > 1 else 0,
            min=min(values),
            max=max(values),
            p50=sorted_vals[int(n * 0.5)],
            p90=sorted_vals[int(n * 0.9)] if n >= 10 else sorted_vals[-1],
            p99=sorted_vals[int(n * 0.99)] if n >= 100 else sorted_vals[-1],
            samples=n,
        )


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    name: str
    device_name: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    iterations: int = 0
    warmup_iterations: int = 0
    
    # Latency metrics
    total_latency: Optional[BenchmarkStats] = None
    vision_latency: Optional[BenchmarkStats] = None
    generation_latency: Optional[BenchmarkStats] = None
    time_to_first_token: Optional[BenchmarkStats] = None
    
    # Throughput metrics
    tokens_per_second: Optional[BenchmarkStats] = None
    images_per_second: Optional[BenchmarkStats] = None
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    
    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_results: List[LatencyResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "device_name": self.device_name,
            "timestamp": self.timestamp,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "peak_memory_mb": self.peak_memory_mb,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "metadata": self.metadata,
        }
        
        for stat_name in ["total_latency", "vision_latency", "generation_latency",
                         "time_to_first_token", "tokens_per_second", "images_per_second"]:
            stat = getattr(self, stat_name)
            if stat:
                result[stat_name] = asdict(stat)
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save results to file."""
        Path(path).write_text(self.to_json())
        logger.info(f"Saved benchmark results to {path}")

    
    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        print(f"\n{'='*60}")
        print(f"Benchmark: {self.name}")
        print(f"Device: {self.device_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Iterations: {self.iterations} (warmup: {self.warmup_iterations})")
        print(f"{'='*60}")
        
        if self.total_latency:
            print(f"\nLatency:")
            print(f"  {self.total_latency}")
            if self.vision_latency:
                print(f"  {self.vision_latency}")
            if self.generation_latency:
                print(f"  {self.generation_latency}")
            if self.time_to_first_token:
                print(f"  {self.time_to_first_token}")
        
        if self.tokens_per_second or self.images_per_second:
            print(f"\nThroughput:")
            if self.tokens_per_second:
                print(f"  {self.tokens_per_second}")
            if self.images_per_second:
                print(f"  {self.images_per_second}")
        
        if self.peak_memory_mb > 0:
            print(f"\nMemory:")
            print(f"  Peak: {self.peak_memory_mb:.1f} MB")
            if self.memory_bandwidth_gbps > 0:
                print(f"  Bandwidth: {self.memory_bandwidth_gbps:.2f} GB/s")
        
        print(f"{'='*60}\n")


class Benchmark:
    """
    Performance benchmark runner for SiLens devices.
    
    Example:
        benchmark = Benchmark(engine)
        
        # Run latency benchmark
        result = benchmark.run_latency(
            image="test.jpg",
            prompt="Describe this image.",
            iterations=100,
            warmup=10,
        )
        
        result.print_summary()
        result.save("benchmark_results.json")
    """
    
    def __init__(
        self,
        engine: "InferenceEngine",
        name: str = "silens-benchmark",
    ):
        """
        Initialize benchmark.
        
        Args:
            engine: InferenceEngine instance
            name: Benchmark name for results
        """
        self.engine = engine
        self.name = name
    
    def run_latency(
        self,
        image: Union[str, Path, np.ndarray],
        prompt: str = "Describe this image in detail.",
        iterations: int = 100,
        warmup: int = 10,
        max_new_tokens: int = 64,
    ) -> BenchmarkResult:
        """
        Run latency benchmark.
        
        Args:
            image: Test image
            prompt: Test prompt
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            max_new_tokens: Max tokens per iteration
            
        Returns:
            BenchmarkResult with latency statistics
        """
        device_name = str(self.engine.device)
        
        # Warmup
        logger.info(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            self.engine.run(image, prompt, max_new_tokens=max_new_tokens)
        
        # Benchmark
        logger.info(f"Running {iterations} benchmark iterations...")
        results: List[LatencyResult] = []
        
        for i in range(iterations):
            result = self.engine.run(image, prompt, max_new_tokens=max_new_tokens)
            
            latency = LatencyResult(
                total_ms=result.total_time_ms,
                vision_ms=result.vision_time_ms,
                generation_ms=result.generation_time_ms,
                preprocessing_ms=result.preprocessing_time_ms,
                tokens_generated=result.num_tokens,
            )
            results.append(latency)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Completed {i + 1}/{iterations}")
        
        # Calculate statistics
        return BenchmarkResult(
            name=self.name,
            device_name=device_name,
            iterations=iterations,
            warmup_iterations=warmup,
            total_latency=BenchmarkStats.from_values(
                [r.total_ms for r in results], "Total Latency", "ms"
            ),
            vision_latency=BenchmarkStats.from_values(
                [r.vision_ms for r in results], "Vision Latency", "ms"
            ),
            generation_latency=BenchmarkStats.from_values(
                [r.generation_ms for r in results], "Generation Latency", "ms"
            ),
            tokens_per_second=BenchmarkStats.from_values(
                [r.tokens_per_second for r in results], "Tokens/sec", "tok/s"
            ),
            images_per_second=BenchmarkStats.from_values(
                [1000.0 / r.total_ms for r in results], "Images/sec", "img/s"
            ),
            raw_results=results,
            metadata={
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            },
        )

    
    def run_throughput(
        self,
        images: List[Union[str, Path, np.ndarray]],
        prompt: str = "Describe this image.",
        max_new_tokens: int = 64,
    ) -> BenchmarkResult:
        """
        Run throughput benchmark with multiple images.
        
        Args:
            images: List of test images
            prompt: Test prompt
            max_new_tokens: Max tokens per image
            
        Returns:
            BenchmarkResult with throughput statistics
        """
        device_name = str(self.engine.device)
        
        logger.info(f"Running throughput benchmark with {len(images)} images...")
        
        start_time = time.time()
        total_tokens = 0
        
        for i, image in enumerate(images):
            result = self.engine.run(image, prompt, max_new_tokens=max_new_tokens)
            total_tokens += result.num_tokens
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(images)} images")
        
        elapsed = time.time() - start_time
        
        return BenchmarkResult(
            name=f"{self.name}-throughput",
            device_name=device_name,
            iterations=len(images),
            warmup_iterations=0,
            tokens_per_second=BenchmarkStats.from_values(
                [total_tokens / elapsed], "Tokens/sec", "tok/s"
            ),
            images_per_second=BenchmarkStats.from_values(
                [len(images) / elapsed], "Images/sec", "img/s"
            ),
            metadata={
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "total_tokens": total_tokens,
                "total_time_sec": elapsed,
            },
        )
    
    def run_memory(
        self,
        image: Union[str, Path, np.ndarray],
        prompt: str = "Describe this image.",
        sequence_lengths: Optional[List[int]] = None,
    ) -> BenchmarkResult:
        """
        Run memory bandwidth benchmark.
        
        Args:
            image: Test image
            prompt: Test prompt
            sequence_lengths: List of sequence lengths to test
            
        Returns:
            BenchmarkResult with memory statistics
        """
        device_name = str(self.engine.device)
        sequence_lengths = sequence_lengths or [64, 128, 256, 512]
        
        logger.info("Running memory benchmark...")
        
        memory_results = []
        
        for seq_len in sequence_lengths:
            result = self.engine.run(image, prompt, max_new_tokens=seq_len)
            
            # Estimate memory bandwidth based on model size and tokens
            # This is a simplified estimate
            model_params = 135_000_000  # 135M params
            bytes_per_param = 0.25  # 2-bit quantized = 0.25 bytes
            model_size_bytes = model_params * bytes_per_param
            
            # KV cache size estimate
            kv_cache_bytes = seq_len * 576 * 30 * 2 * 1  # seq * dim * layers * 2(kv) * bytes
            
            total_bytes = model_size_bytes + kv_cache_bytes
            bandwidth_gbps = (total_bytes * result.num_tokens) / (result.generation_time_ms / 1000) / 1e9
            
            memory_results.append({
                "seq_len": seq_len,
                "bandwidth_gbps": bandwidth_gbps,
            })
        
        avg_bandwidth = statistics.mean([r["bandwidth_gbps"] for r in memory_results])
        
        return BenchmarkResult(
            name=f"{self.name}-memory",
            device_name=device_name,
            iterations=len(sequence_lengths),
            memory_bandwidth_gbps=avg_bandwidth,
            metadata={
                "sequence_lengths": sequence_lengths,
                "results": memory_results,
            },
        )


class ComparisonBenchmark:
    """
    Compare SiLens performance against CPU/GPU baselines.
    
    Example:
        comparison = ComparisonBenchmark(silens_engine)
        
        # Add baseline implementations
        comparison.add_baseline("cpu", cpu_inference_fn)
        comparison.add_baseline("gpu", gpu_inference_fn)
        
        # Run comparison
        results = comparison.run(image, prompt, iterations=50)
        comparison.print_comparison(results)
    """
    
    def __init__(self, silens_engine: "InferenceEngine"):
        self.silens_engine = silens_engine
        self.baselines: Dict[str, Callable] = {}
    
    def add_baseline(
        self,
        name: str,
        inference_fn: Callable[[Any, str], Dict[str, Any]],
    ) -> None:
        """
        Add a baseline implementation.
        
        The inference function should accept (image, prompt) and return
        a dict with keys: text, num_tokens, time_ms
        """
        self.baselines[name] = inference_fn
    
    def run(
        self,
        image: Union[str, Path, np.ndarray],
        prompt: str = "Describe this image.",
        iterations: int = 50,
        warmup: int = 5,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run comparison benchmark.
        
        Returns:
            Dict mapping device name to BenchmarkResult
        """
        results = {}
        
        # Benchmark SiLens
        logger.info("Benchmarking SiLens...")
        benchmark = Benchmark(self.silens_engine, "silens")
        results["silens"] = benchmark.run_latency(
            image, prompt, iterations=iterations, warmup=warmup
        )
        
        # Benchmark baselines
        for name, inference_fn in self.baselines.items():
            logger.info(f"Benchmarking {name}...")
            
            # Warmup
            for _ in range(warmup):
                inference_fn(image, prompt)
            
            # Benchmark
            latencies = []
            tokens = []
            
            for _ in range(iterations):
                start = time.time()
                result = inference_fn(image, prompt)
                elapsed_ms = (time.time() - start) * 1000
                
                latencies.append(elapsed_ms)
                tokens.append(result.get("num_tokens", 0))
            
            results[name] = BenchmarkResult(
                name=name,
                device_name=name,
                iterations=iterations,
                warmup_iterations=warmup,
                total_latency=BenchmarkStats.from_values(
                    latencies, "Total Latency", "ms"
                ),
                tokens_per_second=BenchmarkStats.from_values(
                    [t / (l / 1000) for t, l in zip(tokens, latencies) if l > 0],
                    "Tokens/sec", "tok/s"
                ),
            )
        
        return results

    
    def print_comparison(self, results: Dict[str, BenchmarkResult]) -> None:
        """Print comparison table."""
        print("\n" + "="*80)
        print("Performance Comparison")
        print("="*80)
        
        # Header
        print(f"{'Device':<15} {'Latency (ms)':<20} {'Tokens/sec':<15} {'Speedup':<10}")
        print("-"*60)
        
        # Get SiLens baseline for speedup calculation
        silens_latency = results.get("silens", {})
        if silens_latency and silens_latency.total_latency:
            base_latency = silens_latency.total_latency.mean
        else:
            base_latency = None
        
        # Print each result
        for name, result in sorted(results.items()):
            latency_str = ""
            tps_str = ""
            speedup_str = ""
            
            if result.total_latency:
                lat = result.total_latency
                latency_str = f"{lat.mean:.1f} ± {lat.std:.1f}"
                
                if base_latency and name != "silens":
                    speedup = lat.mean / base_latency
                    speedup_str = f"{speedup:.2f}x"
                elif name == "silens":
                    speedup_str = "1.00x (ref)"
            
            if result.tokens_per_second:
                tps_str = f"{result.tokens_per_second.mean:.1f}"
            
            print(f"{name:<15} {latency_str:<20} {tps_str:<15} {speedup_str:<10}")
        
        print("="*80 + "\n")


# Convenience functions

def quick_benchmark(
    engine: "InferenceEngine",
    image: Union[str, Path, np.ndarray],
    iterations: int = 10,
) -> BenchmarkResult:
    """
    Run a quick benchmark with default settings.
    
    Args:
        engine: InferenceEngine instance
        image: Test image
        iterations: Number of iterations
        
    Returns:
        BenchmarkResult
    """
    benchmark = Benchmark(engine, "quick-benchmark")
    return benchmark.run_latency(
        image,
        prompt="Describe this image briefly.",
        iterations=iterations,
        warmup=2,
        max_new_tokens=32,
    )


def generate_test_image(size: int = 384) -> np.ndarray:
    """
    Generate a random test image for benchmarking.
    
    Args:
        size: Image size (square)
        
    Returns:
        Random RGB image as numpy array
    """
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def load_benchmark_suite(
    images_dir: Union[str, Path],
    max_images: int = 100,
) -> List[np.ndarray]:
    """
    Load a suite of test images from a directory.
    
    Args:
        images_dir: Directory containing test images
        max_images: Maximum number of images to load
        
    Returns:
        List of loaded images as numpy arrays
    """
    from silens.utils import load_image
    
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    images = []
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() in extensions:
            try:
                images.append(load_image(str(img_path)))
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
        
        if len(images) >= max_images:
            break
    
    logger.info(f"Loaded {len(images)} test images from {images_dir}")
    return images
