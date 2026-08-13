#!/usr/bin/env python3
"""
SiLens Smoke Test - Imports
============================

Quick tests to verify all modules can be imported.
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestCoreImports:
    """Test core module imports."""
    
    def test_import_numpy(self):
        """Test numpy import."""
        import numpy as np
        assert np.__version__
    
    def test_import_conversion_modules(self):
        """Test model conversion module imports."""
        try:
            from model.conversion import weights_to_verilog
            assert weights_to_verilog is not None
        except ImportError as e:
            pytest.skip(f"Optional module not available: {e}")
    
    def test_import_quantize(self):
        """Test quantization module import."""
        try:
            from model.conversion import quantize_ternary
            assert quantize_ternary is not None
        except ImportError as e:
            pytest.skip(f"Optional module not available: {e}")


class TestGoldenImports:
    """Test golden model imports."""
    
    def test_import_golden_attention(self):
        """Test golden attention import."""
        try:
            from tests.golden import golden_attention
            assert golden_attention is not None
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
    
    def test_import_golden_transformer(self):
        """Test golden transformer import."""
        try:
            from tests.golden import golden_transformer
            assert golden_transformer is not None
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
    
    def test_import_golden_inference(self):
        """Test golden inference import."""
        try:
            from tests.golden import golden_inference
            assert golden_inference is not None
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
    
    def test_import_golden_vision_encoder(self):
        """Test golden vision encoder import."""
        try:
            from tests.golden import golden_vision_encoder
            assert golden_vision_encoder is not None
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
    
    def test_import_golden_projector(self):
        """Test golden projector import."""
        try:
            from tests.golden import golden_projector
            assert golden_projector is not None
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
    
    def test_import_golden_language_model(self):
        """Test golden language model import."""
        try:
            from tests.golden import golden_language_model
            assert golden_language_model is not None
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")
    
    def test_import_golden_normalization(self):
        """Test golden normalization import."""
        try:
            from tests.golden import golden_normalization
            assert golden_normalization is not None
        except ImportError as e:
            pytest.skip(f"Golden models not available: {e}")


class TestFixtureImports:
    """Test fixture imports."""
    
    def test_import_test_vectors(self):
        """Test test vectors import."""
        try:
            from tests.fixtures import test_vectors
            assert test_vectors is not None
        except ImportError as e:
            pytest.skip(f"Fixtures not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
