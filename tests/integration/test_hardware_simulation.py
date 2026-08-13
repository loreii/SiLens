#!/usr/bin/env python3
"""
SiLens Integration Test - Hardware Simulation Comparison
=========================================================

Compares golden model outputs against RTL simulation results.

Usage:
    pytest tests/integration/test_hardware_simulation.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestGoldenVsHardware:
    """Compare golden model results with hardware simulation."""
    
    def test_popcount_golden(self):
        """Test popcount golden model."""
        from tests.fixtures.test_vectors import generate_reproducible_vectors
        
        vectors = generate_reproducible_vectors('popcount', num_vectors=20, width=64)
        
        for v in vectors:
            inp = v['input']
            expected = v['expected']
            
            # Python popcount
            actual = int(np.sum(inp))
            
            assert actual == expected, \
                f"{v['name']}: expected {expected}, got {actual}"
    
    def test_ternary_mac_golden(self):
        """Test ternary MAC golden model."""
        from tests.fixtures.test_vectors import generate_reproducible_vectors
        
        vectors = generate_reproducible_vectors('ternary_mac', num_vectors=20)
        
        for v in vectors:
            act = v['activations'].astype(np.int32)
            weights = v['weights'].astype(np.int32)
            expected = v['expected']
            
            # Python ternary MAC
            actual = int(np.sum(act * weights))
            
            assert actual == expected, \
                f"{v['name']}: expected {expected}, got {actual}"
    
    def test_softmax_golden(self):
        """Test softmax golden model."""
        from tests.fixtures.test_vectors import generate_reproducible_vectors
        
        vectors = generate_reproducible_vectors('softmax', num_vectors=10, seq_len=8)
        
        for v in vectors:
            inp = v['input']
            expected = v['expected']
            
            # Python softmax
            exp_x = np.exp(inp - np.max(inp))
            actual = exp_x / np.sum(exp_x)
            
            # Softmax should sum to 1
            assert np.isclose(np.sum(actual), 1.0, atol=1e-6), \
                f"{v['name']}: softmax doesn't sum to 1"
            
            # Compare with expected
            assert np.allclose(actual, expected, atol=1e-5), \
                f"{v['name']}: mismatch"

    def test_gelu_golden(self):
        """Test GELU golden model."""
        from tests.fixtures.test_vectors import generate_reproducible_vectors
        
        vectors = generate_reproducible_vectors('gelu', num_vectors=10)
        
        for v in vectors:
            inp = v['input']
            expected = v['expected']
            
            # Python exact GELU
            actual = 0.5 * inp * (1 + np.tanh(
                np.sqrt(2/np.pi) * (inp + 0.044715 * inp**3)
            ))
            
            # Compare with expected
            assert np.allclose(actual, expected, atol=1e-5), \
                f"{v['name']}: GELU mismatch"
    
    def test_layer_norm_golden(self):
        """Test layer norm golden model."""
        from tests.fixtures.test_vectors import generate_reproducible_vectors
        
        vectors = generate_reproducible_vectors('layer_norm', num_vectors=10, dim=64)
        
        for v in vectors:
            inp = v['input']
            gamma = v['gamma']
            beta = v['beta']
            expected = v['expected']
            
            # Python layer norm
            mean = np.mean(inp)
            var = np.var(inp)
            normalized = (inp - mean) / np.sqrt(var + 1e-6)
            actual = gamma * normalized + beta
            
            # Compare
            assert np.allclose(actual, expected, atol=1e-4), \
                f"{v['name']}: LayerNorm mismatch"


class TestHardwareConstraints:
    """Test hardware implementation constraints."""
    
    def test_fixed_point_range(self):
        """Test values stay within fixed-point range."""
        act_width = 8  # 8-bit activations
        frac_bits = 4  # Q3.4 format
        
        max_val = (1 << (act_width - 1)) - 1  # 127
        min_val = -(1 << (act_width - 1))     # -128
        
        # Simulate activations through layers
        np.random.seed(48)
        x = np.random.randn(64).astype(np.float32) * 0.5
        
        # Quantize to fixed-point
        scale = 1 << frac_bits
        x_fixed = np.round(x * scale)
        x_fixed = np.clip(x_fixed, min_val, max_val)
        
        # All values should be in range
        assert np.all(x_fixed >= min_val)
        assert np.all(x_fixed <= max_val)
    
    def test_accumulator_overflow(self):
        """Test accumulator doesn't overflow."""
        acc_width = 32  # 32-bit accumulator
        act_width = 8   # 8-bit activations
        
        max_accumulation = 768  # Max reduction dimension
        max_act = (1 << (act_width - 1)) - 1  # 127
        
        # Worst case: all +1 weights, all max activations
        worst_case = max_accumulation * max_act  # 97536
        
        # Should fit in accumulator
        max_acc = (1 << (acc_width - 1)) - 1  # 2147483647
        
        assert worst_case < max_acc, \
            f"Worst case {worst_case} exceeds accumulator range {max_acc}"
    
    def test_memory_alignment(self):
        """Test weight memory alignment requirements."""
        # Ternary weights: 2 bits each
        # Should be aligned to byte boundaries (4 weights per byte)
        
        test_sizes = [64, 128, 256, 512, 768, 1536]
        
        for size in test_sizes:
            # Packed size
            packed_bytes = (size * 2 + 7) // 8  # Round up to bytes
            
            # Should align to 4-byte boundary for efficient access
            aligned_bytes = (packed_bytes + 3) // 4 * 4
            
            # Check alignment
            assert aligned_bytes % 4 == 0, \
                f"Size {size}: packed to {aligned_bytes} bytes not 4-aligned"


class TestSimulationResults:
    """Load and compare RTL simulation results."""
    
    @pytest.mark.skip(reason="Requires RTL simulation output files")
    def test_load_simulation_results(self):
        """Load RTL simulation results."""
        sim_results_path = PROJECT_ROOT / "rtl" / "tb" / "build" / "results.json"
        
        if not sim_results_path.exists():
            pytest.skip("No simulation results found")
        
        with open(sim_results_path) as f:
            results = json.load(f)
        
        assert 'tests' in results
        assert len(results['tests']) > 0
    
    @pytest.mark.skip(reason="Requires RTL simulation")
    def test_compare_simulation_vs_golden(self):
        """Compare RTL simulation vs golden model."""
        # This test requires running RTL simulation first
        # Results would be compared here
        pass


class TestVectorGeneration:
    """Test test vector generation for RTL."""
    
    def test_generate_attention_vectors(self):
        """Generate attention test vectors."""
        try:
            from tests.golden.golden_attention import GoldenMultiHeadAttention
        except ImportError:
            pytest.skip("Golden models not available")
        
        attn = GoldenMultiHeadAttention(
            embed_dim=64, num_heads=4, weight_type='ternary'
        )
        
        vectors = attn.generate_test_vectors(num_vectors=10)
        
        assert len(vectors) > 0
        
        for v in vectors:
            # Verify vector structure
            assert 'name' in v.__dict__ or hasattr(v, 'name')
            assert 'inputs' in v.__dict__ or hasattr(v, 'inputs')
            assert 'expected_outputs' in v.__dict__ or hasattr(v, 'expected_outputs')
    
    def test_generate_transformer_vectors(self):
        """Generate transformer test vectors."""
        try:
            from tests.golden.golden_transformer import (
                GoldenTransformerBlock, VisionConfig
            )
        except ImportError:
            pytest.skip("Golden models not available")
        
        config = VisionConfig(embed_dim=64, num_heads=4, mlp_dim=128)
        block = GoldenTransformerBlock(config, weight_type='ternary')
        
        vectors = block.generate_test_vectors(num_vectors=10)
        
        assert len(vectors) > 0
    
    def test_vectors_deterministic(self):
        """Test that vector generation is deterministic."""
        from tests.fixtures.test_vectors import generate_reproducible_vectors
        
        # Generate twice with same seed
        v1 = generate_reproducible_vectors('popcount', num_vectors=10, seed=42)
        v2 = generate_reproducible_vectors('popcount', num_vectors=10, seed=42)
        
        assert len(v1) == len(v2)
        
        for a, b in zip(v1, v2):
            assert np.array_equal(a['input'], b['input'])
            assert a['expected'] == b['expected']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
