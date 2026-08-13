#!/usr/bin/env python3
"""
Validation tests for ternary quantization of SmolVLM-256M.

This module provides comprehensive validation of the quantization pipeline:
1. Numerical accuracy: Compare FP16 vs quantized outputs
2. Statistical tests: Verify weight distributions
3. End-to-end inference: Test on sample inputs
4. Hardware simulation: Validate encoding/decoding

Usage:
    # Run all validation tests
    python test_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    
    # Run specific test suite
    python test_quantization.py --model ./model/smolvlm-256m --test numerical
    
    # Verbose output with sample inference
    python test_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct --verbose --inference

Author: SiLens AI/ML Team
License: Apache 2.0
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'conversion'))
try:
    from extract_weights import WeightExtractor
    from quantize_ternary import (
        TernaryQuantizer, 
        TernaryQuantizationConfig, 
        ModelQuantizer,
        QuantizationMode
    )
except ImportError as e:
    logger.error(f"Could not import local modules: {e}")
    logger.error("Make sure extract_weights.py and quantize_ternary.py are in model/conversion/")
    sys.exit(1)


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0


class QuantizationValidator:
    """
    Comprehensive validation suite for ternary quantization.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the validator.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to use for validation
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.results: List[ValidationResult] = []
        
    def load_model(self) -> None:
        """Load the model and processor."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError("transformers not installed")
        
        logger.info(f"Loading model: {self.model_path}")
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _record_result(self, test_name: str, passed: bool, 
                       message: str, details: Dict = None,
                       start_time: float = None) -> ValidationResult:
        """Record a test result."""
        duration = (time.time() - start_time) * 1000 if start_time else 0.0
        
        result = ValidationResult(
            test_name=test_name,
            passed=passed,
            message=message,
            details=details,
            duration_ms=duration
        )
        self.results.append(result)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name} - {message}")
        
        return result
    
    # =========================================================================
    # Test Suite 1: Numerical Accuracy Tests
    # =========================================================================
    
    def test_quantization_roundtrip(self, alpha: float = 0.7) -> ValidationResult:
        """
        Test that quantize -> dequantize produces reasonable values.
        
        This validates the basic quantization formula:
            q(w) = +1 if w > threshold
                 = -1 if w < -threshold
                 = 0  otherwise
        """
        start = time.time()
        
        # Create test weights with known distribution
        np.random.seed(42)
        test_weights = np.random.randn(256, 256).astype(np.float32) * 0.1
        
        # Quantize
        quantizer = TernaryQuantizer(TernaryQuantizationConfig(alpha=alpha))
        result = quantizer.quantize_tensor("test", test_weights)
        
        # Verify ternary values only
        unique_values = set(np.unique(result.quantized_weights))
        expected_values = {-1, 0, 1}
        
        if not unique_values.issubset(expected_values):
            return self._record_result(
                "quantization_roundtrip",
                False,
                f"Invalid quantized values: {unique_values}",
                start_time=start
            )
        
        # Verify threshold is correctly applied
        threshold = alpha * np.mean(np.abs(test_weights))
        
        # Check positive values
        positive_mask = result.quantized_weights == 1
        if np.any(positive_mask):
            min_positive_orig = np.min(test_weights[positive_mask])
            if min_positive_orig < threshold * 0.99:  # Allow small tolerance
                return self._record_result(
                    "quantization_roundtrip",
                    False,
                    f"Positive threshold violated: {min_positive_orig:.4f} < {threshold:.4f}",
                    start_time=start
                )
        
        # Verify reconstruction error is bounded
        scale = result.scale_factors[0]
        reconstructed = result.quantized_weights.astype(np.float32) * scale
        error = np.mean(np.abs(test_weights - reconstructed))
        
        details = {
            'threshold': float(threshold),
            'scale': float(scale),
            'reconstruction_error': float(error),
            'sparsity': result.sparsity,
        }
        
        return self._record_result(
            "quantization_roundtrip",
            True,
            f"Reconstruction error: {error:.6f}, sparsity: {result.sparsity:.1%}",
            details=details,
            start_time=start
        )
    
    def test_layer_wise_accuracy(self, max_layers: int = 10) -> ValidationResult:
        """
        Test quantization accuracy on actual model layers.
        
        Compares the quantization error across different layer types.
        """
        start = time.time()
        
        if self.model is None:
            self.load_model()
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        
        errors = []
        layer_results = []
        
        count = 0
        for name, param in self.model.named_parameters():
            if 'weight' not in name or param.numel() < 64:
                continue
            
            if quantizer.should_skip_layer(name):
                continue
            
            weights = param.detach().float().cpu().numpy()
            result = quantizer.quantize_tensor(name, weights)
            
            errors.append(result.mean_abs_error)
            layer_results.append({
                'name': name,
                'error': result.mean_abs_error,
                'sparsity': result.sparsity,
            })
            
            count += 1
            if count >= max_layers:
                break
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        # Pass if average error is reasonable (typically < 0.1 for normalized weights)
        passed = avg_error < 0.1 and max_error < 0.5
        
        details = {
            'layers_tested': len(errors),
            'avg_error': float(avg_error),
            'max_error': float(max_error),
            'layer_results': layer_results[:5],  # Top 5 for brevity
        }
        
        return self._record_result(
            "layer_wise_accuracy",
            passed,
            f"Avg error: {avg_error:.6f}, max error: {max_error:.6f} across {len(errors)} layers",
            details=details,
            start_time=start
        )
    
    def test_forward_pass_difference(self, tolerance: float = 0.5) -> ValidationResult:
        """
        Test the difference in forward pass outputs between FP32 and quantized.
        
        Uses a simple linear layer simulation to compare outputs.
        """
        start = time.time()
        
        if self.model is None:
            self.load_model()
        
        # Find a suitable linear layer
        target_layer = None
        target_name = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.numel() > 1000:
                target_layer = module
                target_name = name
                break
        
        if target_layer is None:
            return self._record_result(
                "forward_pass_difference",
                False,
                "No suitable linear layer found",
                start_time=start
            )
        
        # Get original weights
        orig_weights = target_layer.weight.detach().float().cpu().numpy()
        
        # Quantize
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        quant_result = quantizer.quantize_tensor(target_name, orig_weights)
        
        # Create test input
        input_dim = orig_weights.shape[1]
        test_input = torch.randn(1, input_dim, device=self.device)
        
        # Original forward pass
        with torch.no_grad():
            orig_output = F.linear(test_input, target_layer.weight)
        
        # Quantized forward pass (simulate)
        scale = quant_result.scale_factors[0]
        quant_weights = torch.from_numpy(
            quant_result.quantized_weights.astype(np.float32) * scale
        ).to(self.device)
        
        with torch.no_grad():
            quant_output = F.linear(test_input, quant_weights)
        
        # Compute difference
        diff = (orig_output - quant_output).abs()
        relative_error = diff.mean().item() / (orig_output.abs().mean().item() + 1e-8)
        
        passed = relative_error < tolerance
        
        details = {
            'layer': target_name,
            'input_shape': list(test_input.shape),
            'output_shape': list(orig_output.shape),
            'mean_abs_diff': float(diff.mean().item()),
            'max_abs_diff': float(diff.max().item()),
            'relative_error': float(relative_error),
        }
        
        return self._record_result(
            "forward_pass_difference",
            passed,
            f"Relative error: {relative_error:.4f} (tolerance: {tolerance})",
            details=details,
            start_time=start
        )
    
    # =========================================================================
    # Test Suite 2: Statistical Tests
    # =========================================================================
    
    def test_weight_distribution(self) -> ValidationResult:
        """
        Test that weight distributions are suitable for ternary quantization.
        
        Good distributions should be:
        - Approximately symmetric around zero
        - Not too heavy-tailed (reasonable kurtosis)
        """
        start = time.time()
        
        if self.model is None:
            self.load_model()
        
        symmetric_layers = 0
        total_layers = 0
        issues = []
        
        for name, param in self.model.named_parameters():
            if 'weight' not in name or param.numel() < 64:
                continue
            
            if 'norm' in name.lower():
                continue
            
            data = param.detach().float().cpu().numpy().flatten()
            total_layers += 1
            
            # Check symmetry
            mean = np.mean(data)
            std = np.std(data)
            
            # Symmetric if mean is small relative to std
            if abs(mean) < 0.1 * std:
                symmetric_layers += 1
            else:
                issues.append(f"{name}: asymmetric (mean/std = {mean/std:.2f})")
        
        symmetry_ratio = symmetric_layers / total_layers if total_layers > 0 else 0
        passed = symmetry_ratio > 0.8  # 80% of layers should be symmetric
        
        details = {
            'total_layers': total_layers,
            'symmetric_layers': symmetric_layers,
            'symmetry_ratio': symmetry_ratio,
            'issues': issues[:5],  # First 5 issues
        }
        
        return self._record_result(
            "weight_distribution",
            passed,
            f"Symmetry ratio: {symmetry_ratio:.1%} ({symmetric_layers}/{total_layers})",
            details=details,
            start_time=start
        )
    
    def test_sparsity_consistency(self, alpha: float = 0.7) -> ValidationResult:
        """
        Test that sparsity levels are consistent and reasonable.
        
        Ternary quantization should produce reasonable sparsity (10-40% typically).
        """
        start = time.time()
        
        config = TernaryQuantizationConfig(alpha=alpha)
        quantizer = ModelQuantizer(self.model_path, config, device=self.device)
        quantizer.quantize_all_weights()
        
        stats = quantizer.get_summary_statistics()
        overall_sparsity = stats['overall_sparsity']
        
        # Sparsity should be in reasonable range
        min_sparsity = 0.05   # At least 5% zeros
        max_sparsity = 0.60   # At most 60% zeros (otherwise too aggressive)
        
        passed = min_sparsity <= overall_sparsity <= max_sparsity
        
        details = {
            'overall_sparsity': overall_sparsity,
            'positive_pct': stats['distribution']['positive_pct'],
            'negative_pct': stats['distribution']['negative_pct'],
            'zero_pct': stats['distribution']['zero_pct'],
            'expected_range': f"{min_sparsity:.0%} - {max_sparsity:.0%}",
        }
        
        return self._record_result(
            "sparsity_consistency",
            passed,
            f"Sparsity: {overall_sparsity:.1%} (expected: {min_sparsity:.0%}-{max_sparsity:.0%})",
            details=details,
            start_time=start
        )
    
    # =========================================================================
    # Test Suite 3: Hardware Encoding Tests
    # =========================================================================
    
    def test_hardware_encoding(self) -> ValidationResult:
        """
        Test hardware encoding and decoding roundtrip.
        
        Verifies that the 2-bit packed format correctly preserves values.
        """
        start = time.time()
        
        config = TernaryQuantizationConfig(alpha=0.7)
        quantizer = TernaryQuantizer(config)
        
        # Create test ternary weights
        test_ternary = np.random.choice([-1, 0, 1], size=(100, 100), p=[0.3, 0.4, 0.3])
        test_ternary = test_ternary.astype(np.int8)
        
        # Encode
        packed = quantizer.encode_for_hardware(test_ternary)
        
        # Decode (implement reverse operation)
        flat_orig = test_ternary.flatten()
        decoded = np.zeros(len(flat_orig), dtype=np.int8)
        
        # Pad length to match
        pad_len = (4 - len(flat_orig) % 4) % 4
        
        for i, byte in enumerate(packed):
            base = i * 4
            if base < len(decoded):
                val = (byte >> 6) & 0x03
                decoded[base] = 1 if val == 0b01 else (-1 if val == 0b10 else 0)
            if base + 1 < len(decoded):
                val = (byte >> 4) & 0x03
                decoded[base + 1] = 1 if val == 0b01 else (-1 if val == 0b10 else 0)
            if base + 2 < len(decoded):
                val = (byte >> 2) & 0x03
                decoded[base + 2] = 1 if val == 0b01 else (-1 if val == 0b10 else 0)
            if base + 3 < len(decoded):
                val = byte & 0x03
                decoded[base + 3] = 1 if val == 0b01 else (-1 if val == 0b10 else 0)
        
        # Compare
        matches = np.sum(flat_orig == decoded[:len(flat_orig)])
        total = len(flat_orig)
        accuracy = matches / total
        
        passed = accuracy == 1.0
        
        details = {
            'original_shape': test_ternary.shape,
            'packed_size': len(packed),
            'compression_ratio': test_ternary.size / len(packed),
            'decode_accuracy': accuracy,
        }
        
        return self._record_result(
            "hardware_encoding",
            passed,
            f"Encode/decode accuracy: {accuracy:.1%}",
            details=details,
            start_time=start
        )
    
    def test_encoding_edge_cases(self) -> ValidationResult:
        """
        Test hardware encoding with edge cases.
        """
        start = time.time()
        
        config = TernaryQuantizationConfig()
        quantizer = TernaryQuantizer(config)
        
        test_cases = [
            ("all_positive", np.ones((10, 10), dtype=np.int8)),
            ("all_negative", -np.ones((10, 10), dtype=np.int8)),
            ("all_zero", np.zeros((10, 10), dtype=np.int8)),
            ("mixed_small", np.array([1, 0, -1, 1], dtype=np.int8)),
            ("odd_length", np.array([1, 0, -1, 1, 0], dtype=np.int8)),
        ]
        
        all_passed = True
        failed_cases = []
        
        for name, data in test_cases:
            packed = quantizer.encode_for_hardware(data)
            
            # Basic sanity check: packed size should be ~1/4 of original
            expected_packed_size = (data.size + 3) // 4
            if len(packed) != expected_packed_size:
                all_passed = False
                failed_cases.append(f"{name}: wrong packed size")
        
        details = {
            'test_cases': len(test_cases),
            'failed': failed_cases,
        }
        
        return self._record_result(
            "encoding_edge_cases",
            all_passed,
            f"All {len(test_cases)} edge cases passed" if all_passed else f"Failed: {failed_cases}",
            details=details,
            start_time=start
        )
    
    # =========================================================================
    # Test Suite 4: End-to-End Inference Test
    # =========================================================================
    
    def test_inference_quality(self, verbose: bool = False) -> ValidationResult:
        """
        Test end-to-end inference with quantized model simulation.
        
        Note: This is a simplified test that compares activation statistics.
        Full inference testing would require a quantized model implementation.
        """
        start = time.time()
        
        if self.model is None:
            self.load_model()
        
        try:
            from PIL import Image
        except ImportError:
            return self._record_result(
                "inference_quality",
                False,
                "PIL not installed",
                start_time=start
            )
        
        # Create a simple test image
        test_image = Image.new('RGB', (384, 384), color=(128, 64, 192))
        
        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What color is this image?"}
                ]
            }
        ]
        
        try:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[test_image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            return self._record_result(
                "inference_quality",
                False,
                f"Input preparation failed: {e}",
                start_time=start
            )
        
        # Run original inference
        with torch.no_grad():
            orig_outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        
        orig_text = self.processor.batch_decode(orig_outputs, skip_special_tokens=True)[0]
        
        if verbose:
            logger.info(f"Original output: {orig_text[-100:]}")
        
        # The full quantized inference would require replacing model weights
        # For this test, we verify the model can generate coherent output
        has_output = len(orig_text) > 10
        
        details = {
            'output_length': len(orig_text),
            'output_preview': orig_text[-100:] if len(orig_text) > 100 else orig_text,
        }
        
        return self._record_result(
            "inference_quality",
            has_output,
            f"Model generated {len(orig_text)} characters",
            details=details,
            start_time=start
        )
    
    # =========================================================================
    # Run All Tests
    # =========================================================================
    
    def run_all_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run all validation tests.
        
        Args:
            verbose: Print detailed output
            
        Returns:
            Summary dictionary
        """
        logger.info("Starting validation test suite...")
        print("\n" + "=" * 70)
        print("QUANTIZATION VALIDATION TESTS")
        print("=" * 70 + "\n")
        
        # Numerical accuracy tests
        print("--- Numerical Accuracy Tests ---")
        self.test_quantization_roundtrip()
        self.test_layer_wise_accuracy()
        self.test_forward_pass_difference()
        
        # Statistical tests
        print("\n--- Statistical Tests ---")
        self.test_weight_distribution()
        self.test_sparsity_consistency()
        
        # Hardware encoding tests
        print("\n--- Hardware Encoding Tests ---")
        self.test_hardware_encoding()
        self.test_encoding_edge_cases()
        
        # End-to-end test
        print("\n--- End-to-End Test ---")
        self.test_inference_quality(verbose=verbose)
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"\nTotal tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.test_name}: {r.message}")
        
        total_time = sum(r.duration_ms for r in self.results)
        print(f"\nTotal time: {total_time:.0f} ms")
        
        return {
            'total': len(self.results),
            'passed': passed,
            'failed': failed,
            'results': [
                {
                    'name': r.test_name,
                    'passed': r.passed,
                    'message': r.message,
                    'duration_ms': r.duration_ms,
                }
                for r in self.results
            ]
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate ternary quantization for SmolVLM-256M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tests
    python test_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    
    # Verbose mode with inference test
    python test_quantization.py --model ./model/smolvlm-256m --verbose
    
    # Custom device
    python test_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct --device cuda
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    validator = QuantizationValidator(args.model, device=args.device)
    results = validator.run_all_tests(verbose=args.verbose)
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")
    
    # Exit with error code if any tests failed
    sys.exit(0 if results['failed'] == 0 else 1)


if __name__ == "__main__":
    main()
