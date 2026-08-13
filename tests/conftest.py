"""
SiLens Test Configuration and Fixtures
======================================

Pytest configuration providing:
- Model loading fixtures
- Weight extraction utilities
- Golden model comparison helpers
- Test data generation

Run tests with:
    pytest tests/ -v
    pytest tests/test_model_loading.py -v
    pytest tests/test_weight_extraction.py -v
"""

import pytest
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Generator
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

# Default model path (can be overridden with --model-path)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "smolvlm-256m"

# Test data directory
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data"

# Fixtures directory
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

# Golden outputs directory
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--model-path",
        action="store",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to SmolVLM model directory"
    )
    parser.addoption(
        "--skip-model-tests",
        action="store_true",
        default=False,
        help="Skip tests that require the model to be downloaded"
    )

    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Update golden reference files instead of comparing"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring downloaded model"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "golden: mark test as golden reference comparison"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options."""
    if config.getoption("--skip-model-tests"):
        skip_model = pytest.mark.skip(reason="--skip-model-tests specified")
        for item in items:
            if "requires_model" in item.keywords:
                item.add_marker(skip_model)


# =============================================================================
# Fixtures - Model Loading
# =============================================================================

@pytest.fixture(scope="session")
def model_path(request) -> Path:
    """Get the model path from command line or default."""
    return Path(request.config.getoption("--model-path"))


@pytest.fixture(scope="session")
def model_available(model_path) -> bool:
    """Check if the model is available."""
    config_file = model_path / "config.json"
    return config_file.exists()


@pytest.fixture(scope="session")
def model_config(model_path, model_available) -> Optional[Dict[str, Any]]:
    """Load model configuration from config.json."""
    if not model_available:
        return None
    
    config_file = model_path / "config.json"
    with open(config_file, 'r') as f:
        return json.load(f)


@pytest.fixture(scope="session")
def hf_model(model_path, model_available):
    """
    Load the HuggingFace model.
    
    This is a session-scoped fixture to avoid reloading
    the model for every test.
    """
    if not model_available:
        pytest.skip("Model not downloaded. Run: make model")
    
    try:
        from transformers import AutoModelForVision2Seq
        import torch
    except ImportError:
        pytest.skip("transformers not installed")
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.eval()
    
    return model


@pytest.fixture(scope="session")
def hf_processor(model_path, model_available):
    """Load the HuggingFace processor."""
    if not model_available:
        pytest.skip("Model not downloaded. Run: make model")
    
    try:
        from transformers import AutoProcessor
    except ImportError:
        pytest.skip("transformers not installed")
    
    return AutoProcessor.from_pretrained(model_path)


# =============================================================================
# Fixtures - Weight Extraction
# =============================================================================

@pytest.fixture(scope="session")
def weight_extractor(hf_model):
    """Create a weight extractor for the model."""
    return WeightExtractor(hf_model)


class WeightExtractor:
    """
    Utility class for extracting and organizing model weights.
    
    Provides methods to:
    - Extract weights by component (vision, projector, LLM)
    - Get weight statistics
    - Export weights in various formats
    """
    
    def __init__(self, model):
        self.model = model
        self._weight_cache = {}
    
    def get_all_weights(self) -> Dict[str, Any]:
        """Get all model weights as a dictionary."""
        if not self._weight_cache:
            for name, param in self.model.named_parameters():
                self._weight_cache[name] = param.detach().cpu()
        return self._weight_cache
    
    def get_vision_weights(self) -> Dict[str, Any]:
        """Get vision encoder weights only."""
        weights = {}
        for name, param in self.model.named_parameters():
            if 'vision' in name.lower() or 'image' in name.lower():
                weights[name] = param.detach().cpu()
        return weights
    
    def get_projector_weights(self) -> Dict[str, Any]:
        """Get multimodal projector weights only."""
        weights = {}
        for name, param in self.model.named_parameters():
            if 'projector' in name.lower():
                weights[name] = param.detach().cpu()
        return weights

    
    def get_llm_weights(self) -> Dict[str, Any]:
        """Get language model weights only."""
        weights = {}
        for name, param in self.model.named_parameters():
            if 'language' in name.lower() or 'model.layers' in name:
                # Exclude projector
                if 'projector' not in name.lower():
                    weights[name] = param.detach().cpu()
        return weights
    
    def get_layer_info(self) -> list:
        """Get information about all layers."""
        layers = []
        for name, param in self.model.named_parameters():
            layers.append({
                'name': name,
                'shape': list(param.shape),
                'numel': param.numel(),
                'dtype': str(param.dtype)
            })
        return layers
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {
            'vision': 0,
            'projector': 0,
            'llm': 0,
            'embeddings': 0,
            'other': 0
        }
        
        for name, param in self.model.named_parameters():
            n = param.numel()
            if 'vision' in name.lower() or 'image' in name.lower():
                counts['vision'] += n
            elif 'projector' in name.lower():
                counts['projector'] += n
            elif 'embed' in name.lower():
                counts['embeddings'] += n
            elif 'language' in name.lower() or 'model.layers' in name:
                counts['llm'] += n
            else:
                counts['other'] += n
        
        counts['total'] = sum(counts.values())
        return counts


# =============================================================================
# Fixtures - Golden Model Comparison
# =============================================================================

@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Get the golden outputs directory."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    return GOLDEN_DIR


@pytest.fixture
def update_golden(request) -> bool:
    """Check if golden files should be updated."""
    return request.config.getoption("--update-golden")


class GoldenComparator:
    """
    Utility for comparing outputs against golden references.
    
    Golden files are stored in tests/golden/ and contain
    reference outputs for regression testing.
    """
    
    def __init__(self, golden_dir: Path):
        self.golden_dir = golden_dir
    
    def save_golden(self, name: str, data: Any) -> Path:
        """Save golden reference data."""
        import numpy as np
        
        filepath = self.golden_dir / f"{name}.npz"
        
        if isinstance(data, dict):
            # Save dict of arrays
            np_data = {k: np.array(v) for k, v in data.items()}
            np.savez(filepath, **np_data)
        else:
            np.savez(filepath, data=np.array(data))
        
        return filepath
    
    def load_golden(self, name: str) -> Optional[Any]:
        """Load golden reference data."""
        import numpy as np
        
        filepath = self.golden_dir / f"{name}.npz"
        
        if not filepath.exists():
            return None
        
        loaded = np.load(filepath, allow_pickle=True)
        
        if 'data' in loaded:
            return loaded['data']
        else:
            return dict(loaded)

    
    def compare(self, name: str, actual: Any, rtol: float = 1e-5, 
                atol: float = 1e-8) -> bool:
        """
        Compare actual output with golden reference.
        
        Args:
            name: Name of the golden file
            actual: Actual output to compare
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            True if outputs match within tolerance
        """
        import numpy as np
        
        golden = self.load_golden(name)
        
        if golden is None:
            raise FileNotFoundError(
                f"Golden file not found: {name}. "
                "Run with --update-golden to create it."
            )
        
        actual_np = np.array(actual)
        
        return np.allclose(actual_np, golden, rtol=rtol, atol=atol)
    
    def compare_or_update(self, name: str, actual: Any, update: bool,
                          rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """
        Compare with golden or update golden file.
        
        Args:
            name: Name of the golden file
            actual: Actual output
            update: If True, update the golden file
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            True if match (or if updated)
        """
        if update:
            self.save_golden(name, actual)
            return True
        
        return self.compare(name, actual, rtol, atol)


@pytest.fixture(scope="session")
def golden_comparator(golden_dir) -> GoldenComparator:
    """Create a golden comparator instance."""
    return GoldenComparator(golden_dir)


# =============================================================================
# Fixtures - Test Data
# =============================================================================

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory."""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Get the fixtures directory."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        pytest.skip("PIL not installed")
    
    # Create a simple test image (RGB, 384x384)
    img_array = np.zeros((384, 384, 3), dtype=np.uint8)
    
    # Add some patterns for testing
    # Red square in top-left
    img_array[0:128, 0:128, 0] = 255
    # Green square in top-right
    img_array[0:128, 256:384, 1] = 255
    # Blue square in bottom-left
    img_array[256:384, 0:128, 2] = 255
    # White square in bottom-right
    img_array[256:384, 256:384, :] = 255
    
    return Image.fromarray(img_array)


@pytest.fixture
def sample_prompt() -> str:
    """Sample prompt for testing."""
    return "What colors do you see in this image?"


# =============================================================================
# Utility Functions
# =============================================================================

def tensor_stats(tensor) -> Dict[str, float]:
    """Get statistics for a tensor."""
    import numpy as np
    
    data = tensor.detach().cpu().numpy().flatten()
    
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'abs_mean': float(np.mean(np.abs(data))),
        'sparsity': float(np.mean(np.abs(data) < 0.01))
    }


def compare_tensors(t1, t2, rtol=1e-5, atol=1e-8) -> bool:
    """Compare two tensors with tolerance."""
    import numpy as np
    
    a1 = t1.detach().cpu().numpy() if hasattr(t1, 'detach') else np.array(t1)
    a2 = t2.detach().cpu().numpy() if hasattr(t2, 'detach') else np.array(t2)
    
    return np.allclose(a1, a2, rtol=rtol, atol=atol)
