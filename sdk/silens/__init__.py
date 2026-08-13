"""
SiLens SDK - Python interface for the SiLens Vision-Language AI Accelerator.

The SiLens SDK provides a high-level Python API for interfacing with SiLens
hardware (PCIe accelerator cards or USB-connected FPGA prototypes) as well
as a simulation mode for development without hardware.

Basic Usage:
    from silens import SiLensDevice, InferenceEngine
    
    # Connect to hardware (or use simulation)
    device = SiLensDevice.discover()[0]  # First available device
    
    # Create inference engine
    engine = InferenceEngine(device)
    
    # Run inference
    result = engine.describe_image("photo.jpg")
    print(result)

For detailed documentation, see: https://github.com/silens/silens-sdk
"""

__version__ = "0.1.0"
__author__ = "SiLens Team"
__license__ = "Apache-2.0"

from silens.device import (
    SiLensDevice,
    SimulatedDevice,
    PCIeDevice,
    USBDevice,
    DeviceError,
    DeviceNotFoundError,
    DeviceTimeoutError,
)
from silens.inference import (
    InferenceEngine,
    InferenceResult,
    StreamingCallback,
)
from silens.model import (
    ModelConfig,
    load_model_config,
)
from silens.utils import (
    load_image,
    preprocess_image,
    Timer,
    MemoryTracker,
)

# New modules
from silens.image_processing import (
    ImageProcessor,
    ImageConfig,
    extract_patches,
    normalize_image,
)
from silens.tokenizer import (
    SiLensTokenizer,
    TokenizerConfig,
    get_tokenizer,
    count_tokens,
)
from silens.streaming import (
    StreamingGenerator,
    StreamingToken,
    StreamingStats,
    StreamingConfig,
    TokenBuffer,
    AsyncInferenceWrapper,
)
from silens.benchmark import (
    Benchmark,
    BenchmarkResult,
    BenchmarkStats,
    ComparisonBenchmark,
    quick_benchmark,
)
from silens.profiler import (
    Profiler,
    ProfileReport,
    ProfileEvent,
    compare_profiles,
)
from silens.multi_device import (
    DevicePool,
    ParallelInference,
    LoadBalancer,
    ParallelStrategy,
    run_distributed_benchmark,
)

__all__ = [
    # Version
    "__version__",
    # Device classes
    "SiLensDevice",
    "SimulatedDevice",
    "PCIeDevice",
    "USBDevice",
    # Errors
    "DeviceError",
    "DeviceNotFoundError",
    "DeviceTimeoutError",
    # Inference
    "InferenceEngine",
    "InferenceResult",
    "StreamingCallback",
    # Model
    "ModelConfig",
    "load_model_config",
    # Utilities
    "load_image",
    "preprocess_image",
    "Timer",
    "MemoryTracker",
    # Image processing
    "ImageProcessor",
    "ImageConfig",
    "extract_patches",
    "normalize_image",
    # Tokenizer
    "SiLensTokenizer",
    "TokenizerConfig",
    "get_tokenizer",
    "count_tokens",
    # Streaming
    "StreamingGenerator",
    "StreamingToken",
    "StreamingStats",
    "StreamingConfig",
    "TokenBuffer",
    "AsyncInferenceWrapper",
    # Benchmark
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkStats",
    "ComparisonBenchmark",
    "quick_benchmark",
    # Profiler
    "Profiler",
    "ProfileReport",
    "ProfileEvent",
    "compare_profiles",
    # Multi-device
    "DevicePool",
    "ParallelInference",
    "LoadBalancer",
    "ParallelStrategy",
    "run_distributed_benchmark",
]
