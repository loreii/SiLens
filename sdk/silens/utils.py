"""
SiLens Utilities.

Helper functions for image loading, preprocessing, timing, and memory management.
"""

from __future__ import annotations

import time
import logging
import functools
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, Callable
from contextlib import contextmanager
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

logger = logging.getLogger(__name__)


# =============================================================================
# Image Loading and Preprocessing
# =============================================================================

def load_image(
    source: Union[str, Path, np.ndarray, "Image.Image"],
    mode: str = "RGB"
) -> np.ndarray:
    """
    Load an image from various sources.
    
    Args:
        source: Image path, URL, numpy array, or PIL Image
        mode: Color mode ("RGB", "L", etc.)
        
    Returns:
        Numpy array with shape (H, W, C) for RGB or (H, W) for grayscale
        
    Raises:
        ValueError: If source type is not supported
        FileNotFoundError: If image file doesn't exist
    """
    # Already numpy array
    if isinstance(source, np.ndarray):
        return source
    
    # PIL Image
    if HAS_PIL and isinstance(source, Image.Image):
        if source.mode != mode:
            source = source.convert(mode)
        return np.array(source)
    
    # File path
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        if not HAS_PIL:
            raise ImportError("PIL is required to load images: pip install pillow")
        
        img = Image.open(path).convert(mode)
        return np.array(img)
    
    raise ValueError(f"Unsupported image source type: {type(source)}")


def preprocess_image(
    image: Union[np.ndarray, "Image.Image"],
    size: int = 384,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    to_uint8: bool = False,
) -> np.ndarray:
    """
    Preprocess image for SiLens hardware.
    
    Performs:
    1. Convert to numpy if PIL Image
    2. Resize to target size (bilinear interpolation)
    3. Optional normalization (ImageNet stats)
    4. Convert to target dtype
    
    Args:
        image: Input image (numpy or PIL)
        size: Target size (square)
        normalize: Whether to normalize pixel values
        mean: Normalization mean (per channel)
        std: Normalization std (per channel)
        to_uint8: Convert to uint8 (0-255) after normalization
        
    Returns:
        Preprocessed image array with shape (size, size, 3)
    """
    # Convert PIL to numpy
    if HAS_PIL and isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    
    # Ensure RGB
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:  # RGBA
        image = image[..., :3]
    
    # Resize
    if image.shape[0] != size or image.shape[1] != size:
        image = _resize_image(image, size)
    
    # Normalize
    if normalize:
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(mean)) / np.array(std)
    
    # Convert to uint8 if requested
    if to_uint8:
        if normalize:
            # Scale normalized values back to 0-255 range
            # Use a simple linear mapping for hardware compatibility
            image = np.clip(image * 64 + 128, 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    return image


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize image using bilinear interpolation."""
    if HAS_PIL:
        pil_img = Image.fromarray(image.astype(np.uint8))
        pil_img = pil_img.resize((size, size), Image.Resampling.BILINEAR)
        return np.array(pil_img)
    else:
        # Simple numpy resize (nearest neighbor as fallback)
        h, w = image.shape[:2]
        y_ratio = h / size
        x_ratio = w / size
        
        y_indices = (np.arange(size) * y_ratio).astype(int)
        x_indices = (np.arange(size) * x_ratio).astype(int)
        
        return image[y_indices[:, None], x_indices]



# =============================================================================
# Timing and Profiling
# =============================================================================

class Timer:
    """
    Simple timing utility for profiling code sections.
    
    Example:
        timer = Timer()
        
        timer.start("preprocessing")
        # ... do preprocessing ...
        timer.stop("preprocessing")
        
        timer.start("inference")
        # ... run inference ...
        timer.stop("inference")
        
        print(timer.summary())
    """
    
    def __init__(self):
        self._starts: Dict[str, float] = {}
        self._times: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
    
    def start(self, name: str) -> None:
        """Start timing a section."""
        self._starts[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing a section and return elapsed time in ms."""
        if name not in self._starts:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = (time.perf_counter() - self._starts[name]) * 1000
        
        if name in self._times:
            self._times[name] += elapsed
            self._counts[name] += 1
        else:
            self._times[name] = elapsed
            self._counts[name] = 1
        
        del self._starts[name]
        return elapsed
    
    def get(self, name: str) -> float:
        """Get total time for a section in ms."""
        return self._times.get(name, 0.0)
    
    def get_avg(self, name: str) -> float:
        """Get average time for a section in ms."""
        if name not in self._times:
            return 0.0
        return self._times[name] / self._counts.get(name, 1)
    
    def total_ms(self) -> float:
        """Get total time across all sections in ms."""
        return sum(self._times.values())
    
    def reset(self) -> None:
        """Reset all timers."""
        self._starts.clear()
        self._times.clear()
        self._counts.clear()
    
    def summary(self) -> str:
        """Get a formatted summary of all timings."""
        lines = ["Timing Summary:"]
        total = self.total_ms()
        
        for name, elapsed in sorted(self._times.items(), key=lambda x: -x[1]):
            count = self._counts[name]
            avg = elapsed / count
            pct = (elapsed / total * 100) if total > 0 else 0
            
            if count > 1:
                lines.append(f"  {name}: {elapsed:.2f}ms total, {avg:.2f}ms avg ({count} calls) [{pct:.1f}%]")
            else:
                lines.append(f"  {name}: {elapsed:.2f}ms [{pct:.1f}%]")
        
        lines.append(f"  TOTAL: {total:.2f}ms")
        return "\n".join(lines)



@contextmanager
def timed(name: str, timer: Optional[Timer] = None):
    """
    Context manager for timing code blocks.
    
    Example:
        timer = Timer()
        with timed("inference", timer):
            result = model.run(image)
    """
    if timer is None:
        timer = Timer()
    
    timer.start(name)
    try:
        yield timer
    finally:
        timer.stop(name)


def timeit(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Example:
        @timeit
        def slow_function():
            time.sleep(1)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__}: {elapsed:.2f}ms")
        return result
    return wrapper


# =============================================================================
# Memory Management
# =============================================================================

class MemoryTracker:
    """
    Track memory usage for debugging and optimization.
    
    Example:
        tracker = MemoryTracker()
        tracker.track("image_buffer", buffer.nbytes)
        tracker.track("kv_cache", cache.nbytes)
        print(tracker.summary())
    """
    
    def __init__(self):
        self._allocations: Dict[str, int] = {}
    
    def track(self, name: str, size: int) -> None:
        """Track a memory allocation."""
        self._allocations[name] = size
    
    def untrack(self, name: str) -> None:
        """Remove tracking for an allocation."""
        self._allocations.pop(name, None)
    
    def get(self, name: str) -> int:
        """Get size of a tracked allocation."""
        return self._allocations.get(name, 0)
    
    def total_bytes(self) -> int:
        """Get total tracked memory in bytes."""
        return sum(self._allocations.values())
    
    def total_mb(self) -> float:
        """Get total tracked memory in megabytes."""
        return self.total_bytes() / (1024 * 1024)
    
    def summary(self) -> str:
        """Get formatted summary of memory usage."""
        lines = ["Memory Usage:"]
        total = self.total_bytes()
        
        for name, size in sorted(self._allocations.items(), key=lambda x: -x[1]):
            mb = size / (1024 * 1024)
            pct = (size / total * 100) if total > 0 else 0
            lines.append(f"  {name}: {mb:.2f} MB [{pct:.1f}%]")
        
        lines.append(f"  TOTAL: {self.total_mb():.2f} MB")
        return "\n".join(lines)


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"



# =============================================================================
# Data Conversion Utilities
# =============================================================================

def quantize_to_ternary(weights: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Quantize floating-point weights to ternary (-1, 0, +1).
    
    Args:
        weights: Float weights to quantize
        threshold: Threshold for zero (values with |w| < threshold*std become 0)
        
    Returns:
        Ternary weights as int8 array
    """
    std = np.std(weights)
    thresh = threshold * std
    
    result = np.zeros_like(weights, dtype=np.int8)
    result[weights > thresh] = 1
    result[weights < -thresh] = -1
    
    return result


def pack_ternary_weights(weights: np.ndarray) -> np.ndarray:
    """
    Pack ternary weights for efficient storage.
    
    Packs 16 ternary values (-1, 0, +1) into 32 bits:
    - 16 bits for sign (1 = negative, 0 = non-negative)
    - 16 bits for zero mask (1 = zero, 0 = non-zero)
    
    Args:
        weights: Ternary weights as int8 array
        
    Returns:
        Packed weights as uint32 array
    """
    if len(weights) % 16 != 0:
        # Pad to multiple of 16
        pad_len = 16 - (len(weights) % 16)
        weights = np.pad(weights, (0, pad_len), constant_values=0)
    
    weights = weights.reshape(-1, 16)
    
    # Create sign bits (1 where weight < 0)
    signs = (weights < 0).astype(np.uint32)
    sign_packed = np.sum(signs * (1 << np.arange(16)), axis=1)
    
    # Create zero bits (1 where weight == 0)
    zeros = (weights == 0).astype(np.uint32)
    zero_packed = np.sum(zeros * (1 << np.arange(16)), axis=1)
    
    # Combine: upper 16 bits = sign, lower 16 bits = zero
    return (sign_packed << 16) | zero_packed


def unpack_ternary_weights(packed: np.ndarray) -> np.ndarray:
    """
    Unpack ternary weights from packed format.
    
    Args:
        packed: Packed weights as uint32 array
        
    Returns:
        Ternary weights as int8 array
    """
    # Extract sign and zero bits
    sign_packed = (packed >> 16).astype(np.uint16)
    zero_packed = (packed & 0xFFFF).astype(np.uint16)
    
    # Unpack to boolean arrays
    signs = ((sign_packed[:, None] >> np.arange(16)) & 1).astype(np.int8)
    zeros = ((zero_packed[:, None] >> np.arange(16)) & 1).astype(np.bool_)
    
    # Reconstruct weights: -1 if sign set, +1 if not zero and not sign, 0 if zero
    weights = np.where(zeros, 0, np.where(signs, -1, 1))
    
    return weights.flatten().astype(np.int8)


# =============================================================================
# Debugging Utilities
# =============================================================================

def hexdump(data: bytes, offset: int = 0, length: int = 256) -> str:
    """Format binary data as hexdump."""
    lines = []
    data = data[offset:offset + length]
    
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{offset + i:08x}  {hex_part:<48}  {ascii_part}")
    
    return "\n".join(lines)
