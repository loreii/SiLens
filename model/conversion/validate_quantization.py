#!/usr/bin/env python3
"""
Quantization validation for SmolVLM-256M ternary weights.

This module validates the quality of ternary quantization by comparing
outputs between the original FP32 model and quantized model across
multiple layers and sample inputs.

Features:
- Layer-by-layer output comparison
- Configurable tolerance thresholds
- Accuracy degradation measurement on sample inputs
- Per-layer and aggregate error statistics
- Visual distribution comparisons

Usage:
    python validate_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    python validate_quantization.py --model ./model/smolvlm-256m --quantized ./quantized
    python validate_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct --detailed

Author: SiLens AI/ML Team
License: Apache 2.0
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

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


@dataclass
class LayerValidationResult:
    """Validation result for a single layer."""
    name: str
    shape: Tuple[int, ...]
    
    # Error metrics
    mean_abs_error: float
    max_abs_error: float
    mean_rel_error: float
    max_rel_error: float
    rmse: float
    
    # Correlation metrics
    cosine_similarity: float
    pearson_correlation: float
    
    # Distribution metrics
    original_mean: float
    quantized_mean: float
    original_std: float
    quantized_std: float
    
    # Quality flags
    within_tolerance: bool
    tolerance_used: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        d = asdict(self)
        d['shape'] = list(d['shape'])
        return d


@dataclass
class ValidationSummary:
    """Overall validation summary."""
    total_layers: int
    layers_within_tolerance: int
    layers_outside_tolerance: int
    
    # Aggregate metrics
    avg_mean_abs_error: float
    avg_cosine_similarity: float
    avg_pearson_correlation: float
    
    # Worst cases
    worst_layer_by_mae: str
    worst_mae: float
    worst_layer_by_cosine: str
    worst_cosine: float
    
    # Quality assessment
    overall_quality: str  # 'excellent', 'good', 'acceptable', 'poor'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class InferenceValidationResult:
    """Result of validating inference on sample inputs."""
    input_type: str  # 'random', 'image', 'text'
    num_samples: int
    
    # Output comparison
    output_mae: float
    output_cosine_similarity: float
    
    # Token-level metrics (for text generation)
    token_match_rate: Optional[float] = None
    top5_match_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weights for validation.
    
    Implements: y = x @ (quantized_weights * scale) + bias
    """
    
    def __init__(self, original_linear: nn.Linear, 
                 quantized_weights: np.ndarray,
                 scale_factors: np.ndarray,
                 alpha: float = 0.7):
        super().__init__()
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Store quantized weights (-1, 0, +1) as int8
        self.register_buffer(
            'quantized_weights',
            torch.from_numpy(quantized_weights).to(torch.int8)
        )
        
        # Store scale factors
        self.register_buffer(
            'scale_factors',
            torch.from_numpy(scale_factors.astype(np.float32))
        )
        
        # Copy bias if present
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.clone())
        else:
            self.bias = None
        
        self.alpha = alpha

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ternary weights."""
        # Dequantize weights: W_deq = Q * scale
        # For per-tensor: single scale
        # For per-channel: scale per output channel
        
        weights_float = self.quantized_weights.float()
        
        if self.scale_factors.numel() == 1:
            # Per-tensor quantization
            weights_dequant = weights_float * self.scale_factors[0]
        else:
            # Per-channel quantization (scale per output channel)
            weights_dequant = weights_float * self.scale_factors.view(-1, 1)
        
        output = F.linear(x, weights_dequant, self.bias)
        return output


class QuantizationValidator:
    """
    Validates ternary quantization quality for SmolVLM-256M.
    
    Compares layer-by-layer outputs and measures accuracy degradation.
    """
    
    def __init__(self, 
                 model_path: str,
                 quantized_path: Optional[str] = None,
                 alpha: float = 0.7,
                 device: str = 'cpu'):
        """
        Initialize validator.
        
        Args:
            model_path: Path to original model or HuggingFace ID
            quantized_path: Path to quantized weights (if already saved)
            alpha: Threshold factor used for quantization
            device: Device to run validation on
        """
        self.model_path = model_path
        self.quantized_path = quantized_path
        self.alpha = alpha
        self.device = device
        
        self.model = None
        self.quantized_weights: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.layer_results: Dict[str, LayerValidationResult] = {}

    
    def load_model(self) -> None:
        """Load the original model."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers torch")
            sys.exit(1)
        
        logger.info(f"Loading model: {self.model_path}")
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=self.device
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {total_params:,} parameters")
    
    def load_quantized_weights(self) -> None:
        """Load pre-computed quantized weights from disk."""
        if self.quantized_path is None:
            logger.info("No quantized path provided, will quantize on-the-fly")
            return
        
        quantized_dir = Path(self.quantized_path)
        if not quantized_dir.exists():
            logger.warning(f"Quantized weights not found: {quantized_dir}")
            return
        
        logger.info(f"Loading quantized weights from: {quantized_dir}")
        
        # Load metadata
        metadata_file = quantized_dir / 'quantization_metadata.json'
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            logger.info(f"Quantization config: alpha={metadata.get('config', {}).get('alpha', 'unknown')}")

        
        # Load weight files
        weights_dir = quantized_dir / 'weights'
        if weights_dir.exists():
            for qfile in weights_dir.glob('*_quantized.npy'):
                base_name = qfile.stem.replace('_quantized', '')
                scale_file = weights_dir / f"{base_name}_scales.npy"
                
                quantized = np.load(qfile)
                scales = np.load(scale_file) if scale_file.exists() else np.array([1.0])
                
                # Reconstruct original layer name
                layer_name = base_name.replace('_', '.')
                self.quantized_weights[layer_name] = (quantized, scales)
        
        logger.info(f"Loaded {len(self.quantized_weights)} quantized weight tensors")
    
    def quantize_layer_weights(self, name: str, 
                               weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize weights for a single layer.
        
        Args:
            name: Layer name
            weights: Original FP32 weights
            
        Returns:
            Tuple of (quantized_weights, scale_factors)
        """
        abs_mean = np.mean(np.abs(weights))
        threshold = self.alpha * abs_mean
        
        # Quantize to ternary
        quantized = np.zeros_like(weights, dtype=np.int8)
        quantized[weights > threshold] = 1
        quantized[weights < -threshold] = -1
        
        # Compute scale factor (average magnitude of non-zero weights)
        nonzero_mask = quantized != 0
        if np.any(nonzero_mask):
            scale = np.mean(np.abs(weights[nonzero_mask]))
        else:
            scale = 1.0
        
        return quantized, np.array([scale], dtype=np.float32)

    
    def validate_layer(self, name: str,
                       original_weights: np.ndarray,
                       quantized_weights: np.ndarray,
                       scale_factors: np.ndarray,
                       tolerance: float = 0.1) -> LayerValidationResult:
        """
        Validate quantization for a single layer.
        
        Args:
            name: Layer name
            original_weights: Original FP32 weights
            quantized_weights: Quantized ternary weights
            scale_factors: Scale factors for dequantization
            tolerance: Acceptable error tolerance
            
        Returns:
            LayerValidationResult with detailed metrics
        """
        # Dequantize for comparison
        if scale_factors.size == 1:
            dequantized = quantized_weights.astype(np.float32) * scale_factors[0]
        else:
            # Per-channel scales
            dequantized = quantized_weights.astype(np.float32) * scale_factors.reshape(-1, 1)
        
        original_flat = original_weights.flatten()
        dequant_flat = dequantized.flatten()
        
        # Error metrics
        abs_error = np.abs(original_flat - dequant_flat)
        mean_abs_error = float(np.mean(abs_error))
        max_abs_error = float(np.max(abs_error))
        
        # Relative error (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = abs_error / (np.abs(original_flat) + 1e-10)
            rel_error = np.where(np.isfinite(rel_error), rel_error, 0)
        mean_rel_error = float(np.mean(rel_error))
        max_rel_error = float(np.min([np.max(rel_error), 1e6]))  # Cap at 1M
        
        # RMSE
        rmse = float(np.sqrt(np.mean((original_flat - dequant_flat) ** 2)))

        
        # Correlation metrics
        # Cosine similarity
        norm_orig = np.linalg.norm(original_flat)
        norm_deq = np.linalg.norm(dequant_flat)
        if norm_orig > 1e-10 and norm_deq > 1e-10:
            cosine_similarity = float(np.dot(original_flat, dequant_flat) / (norm_orig * norm_deq))
        else:
            cosine_similarity = 0.0
        
        # Pearson correlation
        if np.std(original_flat) > 1e-10 and np.std(dequant_flat) > 1e-10:
            pearson_correlation = float(np.corrcoef(original_flat, dequant_flat)[0, 1])
        else:
            pearson_correlation = 0.0
        
        # Distribution stats
        original_mean = float(np.mean(original_flat))
        quantized_mean = float(np.mean(dequant_flat))
        original_std = float(np.std(original_flat))
        quantized_std = float(np.std(dequant_flat))
        
        # Tolerance check (based on normalized error)
        normalized_error = mean_abs_error / (np.abs(original_mean) + original_std + 1e-10)
        within_tolerance = normalized_error < tolerance
        
        result = LayerValidationResult(
            name=name,
            shape=tuple(original_weights.shape),
            mean_abs_error=mean_abs_error,
            max_abs_error=max_abs_error,
            mean_rel_error=mean_rel_error,
            max_rel_error=max_rel_error,
            rmse=rmse,
            cosine_similarity=cosine_similarity,
            pearson_correlation=pearson_correlation,
            original_mean=original_mean,
            quantized_mean=quantized_mean,
            original_std=original_std,
            quantized_std=quantized_std,
            within_tolerance=within_tolerance,
            tolerance_used=tolerance
        )
        
        return result

    
    def validate_all_layers(self, tolerance: float = 0.1) -> Dict[str, LayerValidationResult]:
        """
        Validate quantization for all applicable layers.
        
        Args:
            tolerance: Error tolerance threshold
            
        Returns:
            Dictionary of validation results by layer name
        """
        if self.model is None:
            self.load_model()
        
        if self.quantized_path and not self.quantized_weights:
            self.load_quantized_weights()
        
        logger.info("Validating layer-by-layer quantization...")
        
        results = {}
        skipped = 0
        
        for name, param in self.model.named_parameters():
            # Skip non-weight parameters
            if 'bias' in name.lower() or param.ndim < 2:
                skipped += 1
                continue
            
            # Skip small tensors
            if param.numel() < 64:
                skipped += 1
                continue
            
            # Skip normalization layers
            if any(x in name.lower() for x in ['layernorm', 'layer_norm', 'rmsnorm']):
                skipped += 1
                continue
            
            original_weights = param.detach().float().cpu().numpy()
            
            # Get or compute quantized weights
            # Try to find matching key with different separators
            layer_key = name.replace('.', '_')
            if layer_key in self.quantized_weights:
                quantized, scales = self.quantized_weights[layer_key]
            elif name in self.quantized_weights:
                quantized, scales = self.quantized_weights[name]
            else:
                # Quantize on-the-fly
                quantized, scales = self.quantize_layer_weights(name, original_weights)
            
            result = self.validate_layer(
                name, original_weights, quantized, scales, tolerance
            )
            results[name] = result
        
        self.layer_results = results
        logger.info(f"Validated {len(results)} layers, skipped {skipped}")
        
        return results

    
    def get_summary(self) -> ValidationSummary:
        """
        Compute summary statistics across all validated layers.
        
        Returns:
            ValidationSummary with aggregate metrics
        """
        if not self.layer_results:
            self.validate_all_layers()
        
        results = list(self.layer_results.values())
        
        layers_within = sum(1 for r in results if r.within_tolerance)
        layers_outside = len(results) - layers_within
        
        avg_mae = np.mean([r.mean_abs_error for r in results])
        avg_cosine = np.mean([r.cosine_similarity for r in results])
        avg_pearson = np.mean([r.pearson_correlation for r in results])
        
        # Find worst layers
        worst_by_mae = max(results, key=lambda r: r.mean_abs_error)
        worst_by_cosine = min(results, key=lambda r: r.cosine_similarity)
        
        # Overall quality assessment
        if avg_cosine > 0.95 and layers_within / len(results) > 0.9:
            quality = 'excellent'
        elif avg_cosine > 0.90 and layers_within / len(results) > 0.8:
            quality = 'good'
        elif avg_cosine > 0.80 and layers_within / len(results) > 0.7:
            quality = 'acceptable'
        else:
            quality = 'poor'
        
        return ValidationSummary(
            total_layers=len(results),
            layers_within_tolerance=layers_within,
            layers_outside_tolerance=layers_outside,
            avg_mean_abs_error=float(avg_mae),
            avg_cosine_similarity=float(avg_cosine),
            avg_pearson_correlation=float(avg_pearson),
            worst_layer_by_mae=worst_by_mae.name,
            worst_mae=worst_by_mae.mean_abs_error,
            worst_layer_by_cosine=worst_by_cosine.name,
            worst_cosine=worst_by_cosine.cosine_similarity,
            overall_quality=quality
        )

    
    def validate_inference(self, num_samples: int = 10) -> InferenceValidationResult:
        """
        Validate model inference with random inputs.
        
        Compares outputs of original vs quantized model on random inputs.
        
        Args:
            num_samples: Number of random samples to test
            
        Returns:
            InferenceValidationResult with comparison metrics
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Validating inference with {num_samples} random samples...")
        
        # We'll compare the language model's hidden states
        # For a proper validation, we compare layer outputs
        
        # Get embedding dimension from model config
        try:
            hidden_size = self.model.config.text_config.hidden_size
        except:
            hidden_size = 576  # SmolVLM-256M default
        
        all_orig_outputs = []
        all_quant_outputs = []
        
        for i in range(num_samples):
            # Create random input (batch_size=1, seq_len=16)
            random_input = torch.randn(1, 16, hidden_size).to(self.device)
            
            # Get first linear layer for comparison
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and module.in_features == hidden_size:
                    # Original output
                    with torch.no_grad():
                        orig_output = module(random_input)
                    
                    # Quantized output
                    weights = module.weight.detach().float().cpu().numpy()
                    quantized, scales = self.quantize_layer_weights(name, weights)
                    
                    # Create quantized layer
                    quant_layer = TernaryLinear(module, quantized, scales, self.alpha)
                    quant_layer.to(self.device)
                    
                    with torch.no_grad():
                        quant_output = quant_layer(random_input)
                    
                    all_orig_outputs.append(orig_output.cpu().numpy().flatten())
                    all_quant_outputs.append(quant_output.cpu().numpy().flatten())
                    break

        
        if not all_orig_outputs:
            logger.warning("No suitable layers found for inference validation")
            return InferenceValidationResult(
                input_type='random',
                num_samples=0,
                output_mae=0.0,
                output_cosine_similarity=0.0
            )
        
        # Compute aggregate metrics
        orig_concat = np.concatenate(all_orig_outputs)
        quant_concat = np.concatenate(all_quant_outputs)
        
        mae = float(np.mean(np.abs(orig_concat - quant_concat)))
        
        norm_orig = np.linalg.norm(orig_concat)
        norm_quant = np.linalg.norm(quant_concat)
        if norm_orig > 1e-10 and norm_quant > 1e-10:
            cosine = float(np.dot(orig_concat, quant_concat) / (norm_orig * norm_quant))
        else:
            cosine = 0.0
        
        return InferenceValidationResult(
            input_type='random',
            num_samples=num_samples,
            output_mae=mae,
            output_cosine_similarity=cosine
        )
    
    def print_report(self, detailed: bool = False) -> None:
        """Print validation report to console."""
        if not self.layer_results:
            self.validate_all_layers()
        
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("QUANTIZATION VALIDATION REPORT")
        print("=" * 70)
        
        print(f"\nModel: {self.model_path}")
        print(f"Alpha (threshold factor): {self.alpha}")
        
        print(f"\n--- Layer Statistics ---")
        print(f"Total layers validated: {summary.total_layers}")
        print(f"Within tolerance: {summary.layers_within_tolerance} ({100*summary.layers_within_tolerance/summary.total_layers:.1f}%)")
        print(f"Outside tolerance: {summary.layers_outside_tolerance}")

        
        print(f"\n--- Error Metrics ---")
        print(f"Average MAE: {summary.avg_mean_abs_error:.6f}")
        print(f"Average Cosine Similarity: {summary.avg_cosine_similarity:.4f}")
        print(f"Average Pearson Correlation: {summary.avg_pearson_correlation:.4f}")
        
        print(f"\n--- Worst Cases ---")
        print(f"Worst by MAE: {summary.worst_layer_by_mae}")
        print(f"  MAE: {summary.worst_mae:.6f}")
        print(f"Worst by Cosine: {summary.worst_layer_by_cosine}")
        print(f"  Cosine: {summary.worst_cosine:.4f}")
        
        print(f"\n--- Overall Assessment ---")
        quality_emoji = {
            'excellent': '✓✓',
            'good': '✓',
            'acceptable': '~',
            'poor': '✗'
        }
        print(f"Quality: {summary.overall_quality.upper()} {quality_emoji[summary.overall_quality]}")
        
        if summary.overall_quality == 'excellent':
            print("  Quantization is excellent. Expected minimal accuracy loss.")
        elif summary.overall_quality == 'good':
            print("  Quantization is good. Expect <5% accuracy degradation.")
        elif summary.overall_quality == 'acceptable':
            print("  Quantization is acceptable. Expect 5-10% accuracy degradation.")
        else:
            print("  Quantization quality is poor. Consider adjusting alpha or using per-channel quantization.")
        
        if detailed:
            print(f"\n--- Detailed Per-Layer Results ---")
            
            # Sort by cosine similarity (worst first)
            sorted_results = sorted(
                self.layer_results.values(),
                key=lambda r: r.cosine_similarity
            )
            
            for r in sorted_results[:20]:  # Show worst 20
                status = "✓" if r.within_tolerance else "✗"
                print(f"\n{status} {r.name}")
                print(f"    Shape: {r.shape}")
                print(f"    MAE: {r.mean_abs_error:.6f}, RMSE: {r.rmse:.6f}")
                print(f"    Cosine: {r.cosine_similarity:.4f}, Pearson: {r.pearson_correlation:.4f}")
                print(f"    Orig μ/σ: {r.original_mean:.4f}/{r.original_std:.4f}")
                print(f"    Quant μ/σ: {r.quantized_mean:.4f}/{r.quantized_std:.4f}")
            
            if len(sorted_results) > 20:
                print(f"\n... and {len(sorted_results) - 20} more layers")

    
    def export_report(self, output_path: str) -> None:
        """
        Export detailed validation report to JSON.
        
        Args:
            output_path: Path to save JSON report
        """
        if not self.layer_results:
            self.validate_all_layers()
        
        summary = self.get_summary()
        
        report = {
            'config': {
                'model_path': self.model_path,
                'quantized_path': self.quantized_path,
                'alpha': self.alpha,
            },
            'summary': summary.to_dict(),
            'layers': {name: r.to_dict() for name, r in self.layer_results.items()}
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report exported to: {output_file}")
    
    def visualize_errors(self, output_path: Optional[str] = None) -> None:
        """
        Visualize quantization errors across layers.
        
        Args:
            output_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping visualization")
            return
        
        if not self.layer_results:
            self.validate_all_layers()
        
        results = list(self.layer_results.values())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        
        # 1. MAE distribution
        maes = [r.mean_abs_error for r in results]
        axes[0, 0].hist(maes, bins=30, color='steelblue', alpha=0.7)
        axes[0, 0].axvline(np.mean(maes), color='red', linestyle='--', label=f'Mean: {np.mean(maes):.4f}')
        axes[0, 0].set_xlabel('Mean Absolute Error')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('MAE Distribution Across Layers')
        axes[0, 0].legend()
        
        # 2. Cosine similarity distribution
        cosines = [r.cosine_similarity for r in results]
        axes[0, 1].hist(cosines, bins=30, color='forestgreen', alpha=0.7)
        axes[0, 1].axvline(np.mean(cosines), color='red', linestyle='--', label=f'Mean: {np.mean(cosines):.4f}')
        axes[0, 1].set_xlabel('Cosine Similarity')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Cosine Similarity Distribution')
        axes[0, 1].legend()
        
        # 3. MAE vs layer size
        sizes = [np.prod(r.shape) for r in results]
        axes[1, 0].scatter(sizes, maes, alpha=0.5, color='coral')
        axes[1, 0].set_xlabel('Layer Size (parameters)')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].set_title('MAE vs Layer Size')
        axes[1, 0].set_xscale('log')
        
        # 4. Original std vs cosine similarity
        stds = [r.original_std for r in results]
        axes[1, 1].scatter(stds, cosines, alpha=0.5, color='purple')
        axes[1, 1].set_xlabel('Original Weight Std')
        axes[1, 1].set_ylabel('Cosine Similarity')
        axes[1, 1].set_title('Std vs Cosine Similarity')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main entry point for quantization validation."""
    parser = argparse.ArgumentParser(
        description="Validate ternary quantization quality for SmolVLM-256M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation (quantizes on-the-fly)
    python validate_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    
    # Validate pre-quantized weights
    python validate_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct \\
        --quantized ./model/weights/quantized
    
    # Detailed report with visualization
    python validate_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct \\
        --detailed --visualize --output ./validation_report.json
    
    # Custom tolerance and alpha
    python validate_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct \\
        --alpha 0.6 --tolerance 0.15
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--quantized",
        type=str,
        default=None,
        help="Path to pre-quantized weights directory"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Threshold factor for quantization (default: 0.7)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Error tolerance threshold (default: 0.1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to run validation on"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-layer results"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate error visualizations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Export validation report to JSON file"
    )
    parser.add_argument(
        "--inference-samples",
        type=int,
        default=10,
        help="Number of samples for inference validation"
    )
    
    args = parser.parse_args()

    
    print("=" * 70)
    print("SiLens Quantization Validator")
    print("=" * 70)
    
    # Initialize validator
    validator = QuantizationValidator(
        model_path=args.model,
        quantized_path=args.quantized,
        alpha=args.alpha,
        device=args.device
    )
    
    # Load model and quantized weights
    validator.load_model()
    if args.quantized:
        validator.load_quantized_weights()
    
    # Run validation
    validator.validate_all_layers(tolerance=args.tolerance)
    
    # Run inference validation
    print("\n--- Inference Validation ---")
    inf_result = validator.validate_inference(num_samples=args.inference_samples)
    print(f"Inference samples: {inf_result.num_samples}")
    print(f"Output MAE: {inf_result.output_mae:.6f}")
    print(f"Output Cosine Similarity: {inf_result.output_cosine_similarity:.4f}")
    
    # Print report
    validator.print_report(detailed=args.detailed)
    
    # Export report if requested
    if args.output:
        validator.export_report(args.output)
    
    # Generate visualizations if requested
    if args.visualize:
        vis_path = args.output.replace('.json', '_plots.png') if args.output else None
        validator.visualize_errors(vis_path)
    
    # Summary
    summary = validator.get_summary()
    
    print("\n" + "=" * 70)
    if summary.overall_quality in ('excellent', 'good'):
        print("✓ Validation PASSED - Quantization quality is acceptable")
    else:
        print("⚠ Validation WARNING - Consider adjusting quantization parameters")
    print("=" * 70)
    
    # Return exit code based on quality
    sys.exit(0 if summary.overall_quality in ('excellent', 'good', 'acceptable') else 1)


if __name__ == "__main__":
    main()
