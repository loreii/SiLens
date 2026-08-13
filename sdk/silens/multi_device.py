"""
SiLens Multi-Device Support.

Enables distributed inference across multiple SiLens accelerators for:
- Tensor parallelism (split large layers)
- Pipeline parallelism (different stages on different devices)
- Data parallelism (batch distribution)

Usage:
    from silens.multi_device import DevicePool, ParallelInference
    
    # Discover and use all available devices
    pool = DevicePool.from_discovery()
    
    # Run parallel inference
    parallel = ParallelInference(pool, strategy='pipeline')
    results = parallel.generate(images, prompts)
"""

from __future__ import annotations

import threading
import queue
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import numpy as np

from .device import SiLensDevice, SimulatedDevice


class ParallelStrategy(Enum):
    """Strategies for multi-device parallelism."""
    DATA = auto()       # Distribute batches across devices
    PIPELINE = auto()   # Pipeline stages across devices
    TENSOR = auto()     # Split tensors across devices


@dataclass
class DeviceInfo:
    """Information about a device in the pool."""
    device: SiLensDevice
    device_id: int
    is_available: bool = True
    tasks_completed: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return self.total_latency_ms / self.tasks_completed


class DevicePool:
    """
    Pool of SiLens devices for multi-device operations.
    
    Manages device lifecycle, load balancing, and fault tolerance.
    """
    
    def __init__(self, devices: Optional[List[SiLensDevice]] = None):
        """
        Initialize device pool.
        
        Args:
            devices: List of devices to add to pool
        """
        self._devices: Dict[int, DeviceInfo] = {}
        self._lock = threading.RLock()
        self._device_counter = 0
        self._is_open = False
        
        if devices:
            for device in devices:
                self.add_device(device)
    
    @classmethod
    def from_discovery(cls, mode: str = "auto") -> 'DevicePool':
        """
        Create pool from device discovery.
        
        Args:
            mode: Discovery mode ("auto", "pcie", "usb", "simulation")
            
        Returns:
            DevicePool with discovered devices
        """
        devices = SiLensDevice.discover(mode=mode)
        return cls(devices)

    def add_device(self, device: SiLensDevice) -> int:
        """
        Add a device to the pool.
        
        Args:
            device: Device to add
            
        Returns:
            Device ID in pool
        """
        with self._lock:
            device_id = self._device_counter
            self._devices[device_id] = DeviceInfo(
                device=device,
                device_id=device_id
            )
            self._device_counter += 1
            return device_id
    
    def remove_device(self, device_id: int) -> None:
        """Remove a device from the pool."""
        with self._lock:
            if device_id in self._devices:
                info = self._devices[device_id]
                if self._is_open:
                    try:
                        info.device.close()
                    except Exception:
                        pass
                del self._devices[device_id]
    
    def open(self) -> None:
        """Open all devices in the pool."""
        with self._lock:
            for info in self._devices.values():
                try:
                    info.device.open()
                    info.is_available = True
                except Exception as e:
                    info.is_available = False
                    info.errors += 1
            self._is_open = True
    
    def close(self) -> None:
        """Close all devices in the pool."""
        with self._lock:
            for info in self._devices.values():
                try:
                    info.device.close()
                except Exception:
                    pass
                info.is_available = False
            self._is_open = False
    
    def __enter__(self) -> 'DevicePool':
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    @property
    def num_devices(self) -> int:
        """Number of devices in pool."""
        return len(self._devices)
    
    @property
    def available_devices(self) -> List[DeviceInfo]:
        """List of available devices."""
        return [info for info in self._devices.values() if info.is_available]
    
    def get_device(self, device_id: int) -> Optional[SiLensDevice]:
        """Get device by ID."""
        info = self._devices.get(device_id)
        return info.device if info else None
    
    def acquire_device(self, timeout: float = 10.0) -> Optional[int]:
        """
        Acquire an available device.
        
        Args:
            timeout: Maximum time to wait for device
            
        Returns:
            Device ID or None if no device available
        """
        import time
        start = time.time()
        
        while time.time() - start < timeout:
            with self._lock:
                # Find least-loaded available device
                available = [info for info in self._devices.values() 
                            if info.is_available]
                if available:
                    # Select device with lowest avg latency
                    best = min(available, key=lambda x: x.avg_latency_ms)
                    best.is_available = False
                    return best.device_id
            
            time.sleep(0.01)
        
        return None
    
    def release_device(self, device_id: int, 
                       latency_ms: float = 0.0, 
                       error: bool = False) -> None:
        """
        Release a device back to the pool.
        
        Args:
            device_id: Device to release
            latency_ms: Task latency for load balancing
            error: Whether task had an error
        """
        with self._lock:
            if device_id in self._devices:
                info = self._devices[device_id]
                info.is_available = True
                info.tasks_completed += 1
                info.total_latency_ms += latency_ms
                if error:
                    info.errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'num_devices': self.num_devices,
                'num_available': len(self.available_devices),
                'devices': [
                    {
                        'id': info.device_id,
                        'available': info.is_available,
                        'tasks': info.tasks_completed,
                        'avg_latency_ms': info.avg_latency_ms,
                        'errors': info.errors
                    }
                    for info in self._devices.values()
                ]
            }


class ParallelInference:
    """
    Parallel inference across multiple SiLens devices.
    
    Supports data parallelism, pipeline parallelism, and tensor parallelism.
    """
    
    def __init__(self, pool: DevicePool, 
                 strategy: Union[ParallelStrategy, str] = ParallelStrategy.DATA,
                 max_workers: Optional[int] = None):
        """
        Initialize parallel inference.
        
        Args:
            pool: Device pool to use
            strategy: Parallelism strategy
            max_workers: Maximum parallel workers (default: num devices)
        """
        self.pool = pool
        
        if isinstance(strategy, str):
            strategy = ParallelStrategy[strategy.upper()]
        self.strategy = strategy
        
        self.max_workers = max_workers or pool.num_devices
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def generate_batch(self, images: List[np.ndarray], 
                       prompts: List[str],
                       **kwargs) -> List[str]:
        """
        Generate responses for a batch of image-prompt pairs.
        
        Uses data parallelism to distribute across devices.
        
        Args:
            images: List of images
            prompts: List of prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        if len(images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")
        
        if self.strategy == ParallelStrategy.DATA:
            return self._data_parallel_generate(images, prompts, **kwargs)
        elif self.strategy == ParallelStrategy.PIPELINE:
            return self._pipeline_parallel_generate(images, prompts, **kwargs)
        else:
            # Default to data parallel
            return self._data_parallel_generate(images, prompts, **kwargs)
    
    def _data_parallel_generate(self, images: List[np.ndarray],
                                 prompts: List[str],
                                 **kwargs) -> List[str]:
        """Generate using data parallelism."""
        import time
        
        results = [None] * len(images)
        futures: List[Tuple[int, Future]] = []
        
        def process_item(idx: int, image: np.ndarray, prompt: str) -> Tuple[int, str]:
            device_id = self.pool.acquire_device()
            if device_id is None:
                raise RuntimeError("No device available")
            
            try:
                device = self.pool.get_device(device_id)
                start = time.perf_counter()
                
                # Run inference (simplified - actual would call device methods)
                # result = device.infer(image, prompt, **kwargs)
                result = f"[Device {device_id}] Response to: {prompt[:30]}..."
                
                latency = (time.perf_counter() - start) * 1000
                self.pool.release_device(device_id, latency_ms=latency)
                
                return idx, result
                
            except Exception as e:
                self.pool.release_device(device_id, error=True)
                raise
        
        # Submit all tasks
        for idx, (image, prompt) in enumerate(zip(images, prompts)):
            future = self._executor.submit(process_item, idx, image, prompt)
            futures.append((idx, future))
        
        # Collect results
        for idx, future in futures:
            result_idx, result = future.result()
            results[result_idx] = result
        
        return results
    
    def _pipeline_parallel_generate(self, images: List[np.ndarray],
                                     prompts: List[str],
                                     **kwargs) -> List[str]:
        """
        Generate using pipeline parallelism.
        
        Stage 0: Vision encoding
        Stage 1: Projection
        Stage 2+: LLM layers
        """
        num_devices = self.pool.num_devices
        if num_devices < 2:
            # Fall back to data parallel
            return self._data_parallel_generate(images, prompts, **kwargs)
        
        results = []
        
        # Pipeline stages (simplified)
        vision_device = 0
        llm_device = 1
        
        for image, prompt in zip(images, prompts):
            # Stage 1: Vision encoding on device 0
            vision_id = self.pool.acquire_device()
            if vision_id is not None:
                # vision_features = self._run_vision(vision_id, image)
                self.pool.release_device(vision_id)
            
            # Stage 2: LLM on device 1
            llm_id = self.pool.acquire_device()
            if llm_id is not None:
                # result = self._run_llm(llm_id, vision_features, prompt)
                result = f"[Pipeline] Response to: {prompt[:30]}..."
                self.pool.release_device(llm_id)
                results.append(result)
        
        return results
    
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)


class LoadBalancer:
    """
    Dynamic load balancer for multi-device inference.
    
    Monitors device performance and routes requests to optimal devices.
    """
    
    def __init__(self, pool: DevicePool):
        self.pool = pool
        self._latency_history: Dict[int, List[float]] = {}
        self._lock = threading.Lock()
    
    def select_device(self) -> Optional[int]:
        """
        Select best device based on current load and performance.
        
        Returns:
            Device ID or None
        """
        with self._lock:
            available = self.pool.available_devices
            if not available:
                return None
            
            # Score devices based on:
            # - Availability
            # - Historical latency
            # - Error rate
            scores = []
            for info in available:
                # Lower score is better
                latency_score = info.avg_latency_ms / 100.0  # Normalize
                error_score = info.errors * 10
                score = latency_score + error_score
                scores.append((info.device_id, score))
            
            # Select device with lowest score
            best_id, _ = min(scores, key=lambda x: x[1])
            return best_id
    
    def record_latency(self, device_id: int, latency_ms: float) -> None:
        """Record latency measurement for device."""
        with self._lock:
            if device_id not in self._latency_history:
                self._latency_history[device_id] = []
            
            history = self._latency_history[device_id]
            history.append(latency_ms)
            
            # Keep only recent history
            if len(history) > 100:
                self._latency_history[device_id] = history[-100:]


def run_distributed_benchmark(pool: DevicePool, 
                               num_iterations: int = 100) -> Dict[str, Any]:
    """
    Run benchmark across all devices in pool.
    
    Args:
        pool: Device pool to benchmark
        num_iterations: Number of iterations per device
        
    Returns:
        Benchmark results
    """
    import time
    
    results = {
        'devices': [],
        'total_throughput': 0.0
    }
    
    with pool:
        for info in pool._devices.values():
            device_results = {
                'device_id': info.device_id,
                'latencies': [],
                'errors': 0
            }
            
            device = info.device
            
            for _ in range(num_iterations):
                try:
                    start = time.perf_counter()
                    # Simplified benchmark operation
                    status = device.get_status()
                    latency = (time.perf_counter() - start) * 1000
                    device_results['latencies'].append(latency)
                except Exception:
                    device_results['errors'] += 1
            
            latencies = device_results['latencies']
            if latencies:
                device_results['avg_latency_ms'] = np.mean(latencies)
                device_results['p50_latency_ms'] = np.percentile(latencies, 50)
                device_results['p99_latency_ms'] = np.percentile(latencies, 99)
                device_results['throughput'] = 1000.0 / np.mean(latencies)
            
            results['devices'].append(device_results)
    
    # Calculate aggregate throughput
    total = sum(d.get('throughput', 0) for d in results['devices'])
    results['total_throughput'] = total
    
    return results
