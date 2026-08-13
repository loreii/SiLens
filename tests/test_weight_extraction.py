"""
SiLens Weight Extraction Tests
==============================

Tests for the weight extraction pipeline that converts
HuggingFace model weights to hardware-friendly formats.

Tests verify:
- Weight extraction by component works correctly
- Weight statistics are within expected ranges
- Quantization-friendliness metrics are computed correctly
- Weight export formats are valid

Run with:
    pytest tests/test_weight_extraction.py -v
"""

import pytest
from pathlib import Path
import numpy as np


# =============================================================================
# Weight Extraction Tests
# =============================================================================

class TestWeightExtraction:
    """Tests for weight extraction functionality."""
    
    @pytest.mark.requires_model
    def test_get_all_weights(self, weight_extractor):
        """Verify all weights can be extracted."""
        weights = weight_extractor.get_all_weights()
        
        assert len(weights) > 0, "No weights extracted"
        
        # Verify weights are tensors with data
        for name, tensor in list(weights.items())[:5]:
            assert tensor.numel() > 0, f"Empty tensor: {name}"
    
    @pytest.mark.requires_model
    def test_get_vision_weights(self, weight_extractor):
        """Verify vision encoder weights can be extracted."""
        weights = weight_extractor.get_vision_weights()
        
        # Vision encoder should have some weights
        # Note: May be empty if model uses different naming
        if len(weights) > 0:
            print(f"\nVision weights: {len(weights)} tensors")
            
            total_params = sum(w.numel() for w in weights.values())
            print(f"Total vision params: {total_params:,}")
    
    @pytest.mark.requires_model
    def test_get_projector_weights(self, weight_extractor):
        """Verify projector weights can be extracted."""
        weights = weight_extractor.get_projector_weights()
        
        if len(weights) > 0:
            print(f"\nProjector weights: {len(weights)} tensors")
            
            total_params = sum(w.numel() for w in weights.values())
            print(f"Total projector params: {total_params:,}")

    
    @pytest.mark.requires_model
    def test_get_llm_weights(self, weight_extractor):
        """Verify language model weights can be extracted."""
        weights = weight_extractor.get_llm_weights()
        
        if len(weights) > 0:
            print(f"\nLLM weights: {len(weights)} tensors")
            
            total_params = sum(w.numel() for w in weights.values())
            print(f"Total LLM params: {total_params:,}")
    
    @pytest.mark.requires_model
    def test_weight_extraction_completeness(self, weight_extractor):
        """Verify all parameters are accounted for."""
        all_weights = weight_extractor.get_all_weights()
        counts = weight_extractor.count_parameters()
        
        # Sum of extracted weights should match total
        extracted_count = sum(w.numel() for w in all_weights.values())
        
        assert extracted_count == counts['total'], \
            f"Extraction incomplete: {extracted_count:,} vs {counts['total']:,}"


# =============================================================================
# Weight Statistics Tests
# =============================================================================

class TestWeightStatistics:
    """Tests for weight statistics computation."""
    
    @pytest.mark.requires_model
    def test_weight_means_reasonable(self, weight_extractor):
        """Verify weight means are near zero (typical for neural nets)."""
        weights = weight_extractor.get_all_weights()
        
        outliers = []
        for name, tensor in weights.items():
            if 'weight' in name and 'norm' not in name:
                mean = tensor.float().mean().item()
                if abs(mean) > 0.5:  # Unusually large mean
                    outliers.append((name, mean))
        
        if outliers:
            print(f"\nWeights with unusual means:")
            for name, mean in outliers[:5]:
                print(f"  {name}: {mean:.4f}")
        
        # Most weights should have small means
        assert len(outliers) < len(weights) * 0.1, \
            "Too many weights with unusual means"
    
    @pytest.mark.requires_model
    def test_weight_stds_reasonable(self, weight_extractor):
        """Verify weight standard deviations are reasonable."""
        weights = weight_extractor.get_all_weights()
        
        stds = []
        for name, tensor in weights.items():
            if 'weight' in name and 'norm' not in name:
                std = tensor.float().std().item()
                stds.append((name, std))
        
        if stds:
            std_values = [s[1] for s in stds]
            print(f"\nWeight std statistics:")
            print(f"  Min: {min(std_values):.6f}")
            print(f"  Max: {max(std_values):.6f}")
            print(f"  Mean: {np.mean(std_values):.6f}")
        
        # Stds should be positive and not too large
        for name, std in stds:
            assert std > 0, f"Zero std for {name}"
            assert std < 10, f"Unusually large std for {name}: {std}"

    
    @pytest.mark.requires_model
    def test_no_nan_or_inf_weights(self, weight_extractor):
        """Verify no NaN or Inf values in weights."""
        weights = weight_extractor.get_all_weights()
        
        for name, tensor in weights.items():
            data = tensor.float()
            
            has_nan = data.isnan().any().item()
            has_inf = data.isinf().any().item()
            
            assert not has_nan, f"NaN values found in {name}"
            assert not has_inf, f"Inf values found in {name}"


# =============================================================================
# Quantization Analysis Tests
# =============================================================================

class TestQuantizationAnalysis:
    """Tests for quantization-friendliness analysis."""
    
    @pytest.mark.requires_model
    def test_weight_distribution_analysis(self, weight_extractor):
        """Analyze weight distributions for quantization."""
        weights = weight_extractor.get_all_weights()
        
        results = {
            'normal_like': 0,
            'sparse': 0,
            'outlier_heavy': 0
        }
        
        for name, tensor in weights.items():
            if 'weight' not in name or 'norm' in name:
                continue
            
            data = tensor.float().cpu().numpy().flatten()
            std = np.std(data)
            
            if std == 0:
                continue
            
            # Check distribution characteristics
            sparsity = np.mean(np.abs(data) < 0.01 * std)
            kurtosis = np.mean(((data - np.mean(data)) / std) ** 4) - 3
            
            if sparsity > 0.5:
                results['sparse'] += 1
            elif kurtosis > 10:
                results['outlier_heavy'] += 1
            else:
                results['normal_like'] += 1
        
        print(f"\nWeight distribution analysis:")
        print(f"  Normal-like: {results['normal_like']}")
        print(f"  Sparse: {results['sparse']}")
        print(f"  Outlier-heavy: {results['outlier_heavy']}")
        
        total = sum(results.values())
        if total > 0:
            # Most weights should be quantization-friendly
            friendly_pct = (results['normal_like'] + results['sparse']) / total
            assert friendly_pct > 0.5, \
                f"Only {friendly_pct:.1%} of weights are quantization-friendly"

    
    @pytest.mark.requires_model
    def test_ternary_quantization_simulation(self, weight_extractor):
        """
        Simulate ternary quantization and measure information loss.
        
        Ternary quantization maps weights to {-1, 0, +1} which is
        what the SiLens hardware uses for efficient computation.
        """
        weights = weight_extractor.get_all_weights()
        
        quantization_errors = []
        
        for name, tensor in weights.items():
            if 'weight' not in name or 'norm' in name:
                continue
            
            data = tensor.float().cpu().numpy().flatten()
            
            # Skip if all zeros
            if np.max(np.abs(data)) == 0:
                continue
            
            # Ternary quantization: threshold-based
            threshold = 0.5 * np.std(data)
            
            # Quantize to {-1, 0, +1}
            quantized = np.zeros_like(data)
            quantized[data > threshold] = 1
            quantized[data < -threshold] = -1
            
            # Scale back
            scale = np.mean(np.abs(data[np.abs(data) > threshold])) \
                if np.any(np.abs(data) > threshold) else 1.0
            quantized_scaled = quantized * scale
            
            # Compute error
            mse = np.mean((data - quantized_scaled) ** 2)
            var = np.var(data)
            
            if var > 0:
                relative_error = mse / var
                quantization_errors.append((name, relative_error))
        
        if quantization_errors:
            errors = [e[1] for e in quantization_errors]
            print(f"\nTernary quantization analysis:")
            print(f"  Mean relative error: {np.mean(errors):.4f}")
            print(f"  Max relative error: {np.max(errors):.4f}")
            
            # Find layers with highest error
            quantization_errors.sort(key=lambda x: -x[1])
            print(f"\n  Highest error layers:")
            for name, err in quantization_errors[:3]:
                print(f"    {name}: {err:.4f}")


# =============================================================================
# Weight Export Tests
# =============================================================================

class TestWeightExport:
    """Tests for weight export functionality."""
    
    @pytest.mark.requires_model
    def test_layer_info_export(self, weight_extractor, tmp_path):
        """Verify layer info can be exported to JSON."""
        import json
        
        layers = weight_extractor.get_layer_info()
        
        output_file = tmp_path / "layers.json"
        with open(output_file, 'w') as f:
            json.dump(layers, f, indent=2)
        
        assert output_file.exists(), "Export file not created"
        
        # Verify can be read back
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        
        assert len(loaded) == len(layers), "Layer count mismatch after export"

    
    @pytest.mark.requires_model
    def test_weight_export_numpy(self, weight_extractor, tmp_path):
        """Verify weights can be exported to numpy format."""
        weights = weight_extractor.get_all_weights()
        
        # Export first few weights
        export_count = min(5, len(weights))
        exported = {}
        
        for i, (name, tensor) in enumerate(weights.items()):
            if i >= export_count:
                break
            
            # Clean name for filename
            clean_name = name.replace('.', '_').replace('/', '_')
            exported[clean_name] = tensor.cpu().numpy()
        
        output_file = tmp_path / "weights_sample.npz"
        np.savez(output_file, **exported)
        
        assert output_file.exists(), "Export file not created"
        
        # Verify can be read back
        loaded = np.load(output_file)
        assert len(loaded.files) == export_count, "Weight count mismatch"
    
    @pytest.mark.requires_model
    def test_ternary_weight_export(self, weight_extractor, tmp_path):
        """
        Test exporting weights in ternary format.
        
        Ternary weights are stored as:
        - sign: 1 bit per weight
        - mask: 1 bit per weight (0 = zero, 1 = non-zero)
        """
        weights = weight_extractor.get_all_weights()
        
        # Pick a sample weight tensor
        sample_name = None
        sample_tensor = None
        
        for name, tensor in weights.items():
            if 'weight' in name and 'norm' not in name:
                if tensor.numel() > 100:  # Non-trivial size
                    sample_name = name
                    sample_tensor = tensor
                    break
        
        if sample_tensor is None:
            pytest.skip("No suitable weight tensor found")
        
        # Convert to ternary
        data = sample_tensor.float().cpu().numpy()
        threshold = 0.5 * np.std(data)
        
        # Create sign and mask arrays
        sign = (data > 0).astype(np.uint8)
        mask = (np.abs(data) > threshold).astype(np.uint8)
        
        # Pack into bits (8 weights per byte)
        def pack_bits(arr):
            """Pack boolean array into bytes."""
            padded = np.pad(arr.flatten(), 
                          (0, 8 - len(arr.flatten()) % 8), 
                          mode='constant')
            return np.packbits(padded.astype(np.uint8))
        
        sign_packed = pack_bits(sign)
        mask_packed = pack_bits(mask)
        
        # Export
        output_file = tmp_path / "ternary_sample.npz"
        np.savez(output_file,
                 shape=data.shape,
                 sign=sign_packed,
                 mask=mask_packed,
                 threshold=threshold)
        
        print(f"\nTernary export for {sample_name}:")
        print(f"  Original size: {data.nbytes:,} bytes")
        print(f"  Packed size: {sign_packed.nbytes + mask_packed.nbytes:,} bytes")
        print(f"  Compression: {data.nbytes / (sign_packed.nbytes + mask_packed.nbytes):.1f}x")


# =============================================================================
# Golden Reference Tests
# =============================================================================

class TestWeightGoldenReference:
    """
    Golden reference tests for weight extraction.
    
    These tests compare extracted weights against saved references
    to catch regressions.
    """
    
    @pytest.mark.requires_model
    @pytest.mark.golden
    def test_weight_checksums(self, weight_extractor, golden_comparator, 
                               update_golden):
        """
        Verify weight checksums match golden reference.
        
        This catches accidental weight modifications or
        loading issues.
        """
        import hashlib
        
        weights = weight_extractor.get_all_weights()
        
        checksums = {}
        for name, tensor in weights.items():
            data = tensor.cpu().numpy().tobytes()
            checksums[name] = hashlib.md5(data).hexdigest()
        
        golden_name = "weight_checksums"
        
        if update_golden:
            golden_comparator.save_golden(golden_name, checksums)
            print(f"\nUpdated golden checksums for {len(checksums)} weights")
            return
        
        golden = golden_comparator.load_golden(golden_name)
        
        if golden is None:
            pytest.skip("Golden file not found. Run with --update-golden")
        
        golden_dict = golden.item() if hasattr(golden, 'item') else dict(golden)
        
        mismatches = []
        for name, checksum in checksums.items():
            if name in golden_dict and golden_dict[name] != checksum:
                mismatches.append(name)
        
        assert len(mismatches) == 0, \
            f"Weight checksums changed for {len(mismatches)} layers: {mismatches[:5]}"


# =============================================================================
# Performance Tests
# =============================================================================

class TestWeightExtractionPerformance:
    """Performance tests for weight extraction."""
    
    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_extraction_speed(self, hf_model):
        """Measure weight extraction speed."""
        import time
        
        start = time.time()
        
        weights = {}
        for name, param in hf_model.named_parameters():
            weights[name] = param.detach().cpu()
        
        elapsed = time.time() - start
        
        total_params = sum(w.numel() for w in weights.values())
        params_per_sec = total_params / elapsed
        
        print(f"\nExtraction performance:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Speed: {params_per_sec/1e6:.1f}M params/sec")
        
        # Should be reasonably fast
        assert elapsed < 60, f"Extraction too slow: {elapsed:.1f}s"
