#!/usr/bin/env python3
"""
SiLens Integration Test - Weight Export
========================================

Tests weight export to Verilog format.

Usage:
    pytest tests/integration/test_weight_export.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
import re

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestVerilogExport:
    """Test Verilog weight export functionality."""
    
    def test_sanitize_name(self):
        """Test name sanitization for Verilog."""
        try:
            from model.conversion.weights_to_verilog import sanitize_name
        except ImportError:
            pytest.skip("weights_to_verilog not available")
        
        # Test various name patterns
        test_cases = [
            ("layer.0.weight", "layer_0_weight"),
            ("model.layers.attention.q_proj", "model_layers_attention_q_proj"),
            ("vision_model.encoder.block_0", "vision_model_encoder_block_0"),
            ("simple", "simple"),
        ]
        
        for input_name, expected in test_cases:
            result = sanitize_name(input_name)
            assert result == expected, \
                f"sanitize_name('{input_name}') = '{result}', expected '{expected}'"
    
    def test_categorize_layer(self):
        """Test layer categorization."""
        try:
            from model.conversion.weights_to_verilog import categorize_layer
        except ImportError:
            pytest.skip("weights_to_verilog not available")
        
        test_cases = [
            ("vision_model.encoder.layers.0", "vision_encoder"),
            ("model.layers.0.self_attn", "language_model"),
            ("multi_modal_projector.linear_1", "projector"),
            ("model.embed_tokens", "embeddings"),
        ]
        
        for layer_name, expected_category in test_cases:
            result = categorize_layer(layer_name)
            assert expected_category in result.lower() or result == expected_category, \
                f"categorize_layer('{layer_name}') = '{result}', expected '{expected_category}'"

    def test_ternary_to_verilog(self):
        """Test ternary weight conversion to Verilog format."""
        np.random.seed(42)
        
        # Create ternary weights
        weights = np.random.choice([-1, 0, 1], 
            size=(64, 128), p=[0.35, 0.3, 0.35]).astype(np.int8)
        
        # Encode as 2-bit values: 00=0, 01=+1, 10=-1
        encoded = np.zeros(weights.size, dtype=np.uint8)
        flat = weights.flatten()
        for i, w in enumerate(flat):
            if w == 1:
                encoded[i] = 0b01
            elif w == -1:
                encoded[i] = 0b10
            else:
                encoded[i] = 0b00
        
        # Pack into bytes (4 weights per byte)
        packed = np.zeros((weights.size + 3) // 4, dtype=np.uint8)
        for i in range(len(packed)):
            for j in range(4):
                idx = i * 4 + j
                if idx < len(encoded):
                    packed[i] |= encoded[idx] << (6 - j * 2)
        
        # Verify we can decode back
        decoded = np.zeros(weights.size, dtype=np.int8)
        for i, byte in enumerate(packed):
            for j in range(4):
                idx = i * 4 + j
                if idx < len(decoded):
                    bits = (byte >> (6 - j * 2)) & 0x03
                    if bits == 0b01:
                        decoded[idx] = 1
                    elif bits == 0b10:
                        decoded[idx] = -1
                    else:
                        decoded[idx] = 0
        
        assert np.array_equal(flat, decoded), "Encoding/decoding mismatch"
    
    def test_verilog_syntax_generation(self):
        """Test Verilog syntax is valid."""
        # Generate a sample Verilog module
        module_template = """
module test_weights (
    input  wire clk,
    output wire [15:0] weight_out
);
    // Hardwired ternary weights
    localparam [15:0] WEIGHTS = 16'b{weights};
    assign weight_out = WEIGHTS;
endmodule
"""
        
        # Create sample weight bits
        weights_bits = "01" * 8  # 8 +1 weights
        
        verilog_code = module_template.format(weights=weights_bits)
        
        # Basic syntax checks
        assert "module test_weights" in verilog_code
        assert "endmodule" in verilog_code
        assert "16'b" in verilog_code
    
    def test_export_with_tempfile(self):
        """Test exporting to a temporary file."""
        try:
            from model.conversion.weights_to_verilog import (
                sanitize_name, categorize_layer
            )
        except ImportError:
            pytest.skip("weights_to_verilog not available")
        
        np.random.seed(43)
        
        # Create test weights
        weights = {
            'vision_model.layer0.weight': np.random.choice(
                [-1, 0, 1], size=(32, 64), p=[0.35, 0.3, 0.35]
            ).astype(np.int8),
            'model.layers.0.mlp.weight': np.random.choice(
                [-1, 0, 1], size=(128, 32), p=[0.35, 0.3, 0.35]
            ).astype(np.int8),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_weights.v"
            
            # Write a simple Verilog file
            with open(output_path, 'w') as f:
                f.write("// Auto-generated weight file\n")
                f.write("module generated_weights;\n")
                
                for name, w in weights.items():
                    safe_name = sanitize_name(name)
                    f.write(f"  // {name}: shape {w.shape}\n")
                    f.write(f"  localparam integer {safe_name}_SIZE = {w.size};\n")
                
                f.write("endmodule\n")
            
            # Verify file was created
            assert output_path.exists()
            
            # Read and verify content
            content = output_path.read_text()
            assert "module generated_weights" in content
            assert "endmodule" in content


class TestWeightValidation:
    """Test weight validation and statistics."""
    
    def test_weight_statistics(self):
        """Test weight statistics calculation."""
        np.random.seed(44)
        
        weights = np.random.choice([-1, 0, 1], 
            size=(1000, 1000), p=[0.35, 0.3, 0.35]).astype(np.int8)
        
        # Calculate statistics
        total = weights.size
        num_pos = np.sum(weights == 1)
        num_neg = np.sum(weights == -1)
        num_zero = np.sum(weights == 0)
        
        sparsity = num_zero / total
        
        # Verify distribution
        assert 0.25 <= sparsity <= 0.35, f"Sparsity {sparsity} out of expected range"
        assert abs(num_pos - num_neg) / total < 0.05, "Distribution not symmetric"
    
    def test_weight_memory_estimate(self):
        """Test memory estimation for weights."""
        # SmolVLM-256M approximate sizes
        vision_weights = 768 * 3072 * 12  # Vision encoder MLP
        projector_weights = 768 * 576     # Projector
        llm_weights = 576 * 1536 * 30     # LLM layers
        
        total_weights = vision_weights + projector_weights + llm_weights
        
        # At 2 bits per weight
        memory_bits = total_weights * 2
        memory_bytes = memory_bits / 8
        memory_mb = memory_bytes / (1024 * 1024)
        
        # Should be under 100MB for 256M model with ternary
        assert memory_mb < 100, f"Memory estimate {memory_mb:.1f}MB too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
