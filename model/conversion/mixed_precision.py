#!/usr/bin/env python3
"""
Mixed-precision ternary quantization for SmolVLM-256M.

This module implements mixed-precision quantization, keeping critical layers
at higher precision (e.g., 4-bit or 8-bit) while using ternary for the majority
of layers. Critical layers are identified based on sensitivity analysis.

Features:
- Automatic identification of critical layers
- Multi-precision support (ternary, 4-bit, 8-bit)
- Configurable precision assignment strategies
- Memory/accuracy trade-off optimization
- Hardware-aware precision selection

Theory:
    Not all layers contribute equally to model accuracy. Some layers are:
    1. Embedding layers - First/last layers, often critical
    2. Attention layers - QKV projections may need higher precision
    3. Normalization-adjacent - Layers before/after normalization
    
    By keeping ~5-10% of weights at higher precision, we can significantly
    improve accuracy with minimal memory impact.

Usage:
    python mixed_precision.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    python mixed_precision.py --model ./model/smolvlm-256m --high-precision-ratio 0.1

Author: SiLens AI/ML Team
License: Apache 2.0
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrecisionLevel(Enum):
    """Supported precision levels for quantization."""
    TERNARY = "ternary"       # 2-bit: {-1, 0, +1}
    INT4 = "int4"             # 4-bit: {-8, ..., +7}
    INT8 = "int8"             # 8-bit: {-128, ..., +127}
    FP16 = "fp16"             # 16-bit floating point
    FP32 = "fp32"             # 32-bit floating point (no quantization)


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed-precision quantization."""
    default_precision: PrecisionLevel = PrecisionLevel.TERNARY
    high_precision_ratio: float = 0.1         # Fraction of layers to keep at high precision
    high_precision_level: PrecisionLevel = PrecisionLevel.INT8
    
    # Layer importance criteria weights
    first_last_layer_weight: float = 2.0
    attention_weight: float = 1.5
    embedding_weight: float = 2.0
    lm_head_weight: float = 2.0
    
    # Automatic selection parameters
    sensitivity_threshold: float = 0.8        # Cosine similarity threshold
    min_layer_size: int = 1024                # Minimum layer size for high precision
    
    # Explicit layer assignments (override automatic selection)
    force_high_precision: List[str] = field(default_factory=list)
    force_ternary: List[str] = field(default_factory=list)


@dataclass
class LayerPrecisionAssignment:
    """Precision assignment for a single layer."""
    name: str
    shape: Tuple[int, ...]
    numel: int
    precision: PrecisionLevel
    importance_score: float
    reason: str                               # Why this precision was assigned
    memory_bytes: int                         # Memory required
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'name': self.name,
            'shape': list(self.shape),
            'numel': self.numel,
            'precision': self.precision.value,
            'importance_score': self.importance_score,
            'reason': self.reason,
            'memory_bytes': self.memory_bytes,
        }


@dataclass
class MixedPrecisionSummary:
    """Summary of mixed-precision assignment."""
    total_layers: int
    precision_counts: Dict[str, int]
    precision_params: Dict[str, int]
    total_memory_bytes: int
    memory_vs_fp32_ratio: float
    high_precision_layers: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_layers': self.total_layers,
            'precision_counts': self.precision_counts,
            'precision_params': self.precision_params,
            'total_memory_bytes': self.total_memory_bytes,
            'memory_vs_fp32_ratio': self.memory_vs_fp32_ratio,
            'high_precision_layers': self.high_precision_layers,
        }


class LayerImportanceScorer:
    """
    Scores layer importance for precision assignment.
    
    Higher scores indicate layers that should be kept at higher precision.
    """
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        
    def _is_first_or_last_layer(self, name: str, all_names: List[str]) -> bool:
        """Check if layer is among first or last layers."""
        # Find layer index in sorted list
        weight_layers = [n for n in all_names if 'weight' in n.lower()]
        if name not in weight_layers:
            return False
        
        idx = weight_layers.index(name)
        total = len(weight_layers)
        
        # First 5% or last 5%
        return idx < total * 0.05 or idx > total * 0.95
    
    def _is_attention_layer(self, name: str) -> bool:
        """Check if layer is part of attention mechanism."""
        name_lower = name.lower()
        return any(x in name_lower for x in [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'query', 'key', 'value',
            'attn', 'attention'
        ])
    
    def _is_embedding_layer(self, name: str) -> bool:
        """Check if layer is an embedding layer."""
        name_lower = name.lower()
        return any(x in name_lower for x in [
            'embed', 'wte', 'wpe', 'token_embedding',
            'position_embedding', 'patch_embed'
        ])
    
    def _is_output_layer(self, name: str) -> bool:
        """Check if layer is the output/head layer."""
        name_lower = name.lower()
        return any(x in name_lower for x in [
            'lm_head', 'classifier', 'output_projection',
            'head.weight', 'linear_head'
        ])
    
    def score_layer(self, name: str, 
                    param: torch.nn.Parameter,
                    all_names: List[str],
                    sensitivity: Optional[float] = None) -> Tuple[float, str]:
        """
        Score a layer's importance.
        
        Args:
            name: Layer name
            param: Layer parameter tensor
            all_names: All layer names in model
            sensitivity: Optional sensitivity score from analysis
            
        Returns:
            Tuple of (importance_score, reason)
        """
        score = 1.0
        reasons = []
        
        # Check special layer types
        if self._is_first_or_last_layer(name, all_names):
            score *= self.config.first_last_layer_weight
            reasons.append("first/last layer")
        
        if self._is_attention_layer(name):
            score *= self.config.attention_weight
            reasons.append("attention layer")
        
        if self._is_embedding_layer(name):
            score *= self.config.embedding_weight
            reasons.append("embedding layer")
        
        if self._is_output_layer(name):
            score *= self.config.lm_head_weight
            reasons.append("output head")
        
        # Sensitivity-based scoring
        if sensitivity is not None and sensitivity < self.config.sensitivity_threshold:
            # Low sensitivity means layer is critical
            score *= (1 + (self.config.sensitivity_threshold - sensitivity))
            reasons.append(f"sensitive (sim={sensitivity:.2f})")
        
        # Layer size consideration (larger layers have more impact)
        if param.numel() > 1_000_000:
            score *= 1.2
            reasons.append("large layer")
        
        reason = ", ".join(reasons) if reasons else "default"
        
        return score, reason


class MixedPrecisionQuantizer:
    """
    Mixed-precision quantizer for neural networks.
    
    Assigns different precision levels to different layers based on
    their importance to model accuracy.
    """
    
    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        """
        Initialize the mixed-precision quantizer.
        
        Args:
            config: Mixed-precision configuration
        """
        self.config = config or MixedPrecisionConfig()
        self.scorer = LayerImportanceScorer(self.config)
        self.assignments: Dict[str, LayerPrecisionAssignment] = {}
        
    def _bytes_per_param(self, precision: PrecisionLevel) -> float:
        """Get bytes per parameter for a precision level."""
        return {
            PrecisionLevel.TERNARY: 0.25,    # 2 bits
            PrecisionLevel.INT4: 0.5,         # 4 bits
            PrecisionLevel.INT8: 1.0,         # 8 bits
            PrecisionLevel.FP16: 2.0,         # 16 bits
            PrecisionLevel.FP32: 4.0,         # 32 bits
        }[precision]
    
    def _quantize_ternary(self, weights: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """Quantize to ternary values."""
        threshold = alpha * np.mean(np.abs(weights))
        quantized = np.zeros_like(weights, dtype=np.int8)
        quantized[weights > threshold] = 1
        quantized[weights < -threshold] = -1
        return quantized
    
    def _quantize_int4(self, weights: np.ndarray) -> np.ndarray:
        """Quantize to 4-bit integers."""
        # Scale to [-8, 7] range
        max_val = np.max(np.abs(weights))
        if max_val > 0:
            scale = 7.0 / max_val
        else:
            scale = 1.0
        
        quantized = np.clip(np.round(weights * scale), -8, 7).astype(np.int8)
        return quantized
    
    def _quantize_int8(self, weights: np.ndarray) -> np.ndarray:
        """Quantize to 8-bit integers."""
        # Scale to [-127, 127] range
        max_val = np.max(np.abs(weights))
        if max_val > 0:
            scale = 127.0 / max_val
        else:
            scale = 1.0
        
        quantized = np.clip(np.round(weights * scale), -127, 127).astype(np.int8)
        return quantized
    
    def quantize_layer(self, weights: np.ndarray, 
                       precision: PrecisionLevel) -> Tuple[np.ndarray, float]:
        """
        Quantize a layer with the specified precision.
        
        Args:
            weights: Original FP32 weights
            precision: Target precision level
            
        Returns:
            Tuple of (quantized_weights, scale_factor)
        """
        if precision == PrecisionLevel.TERNARY:
            quantized = self._quantize_ternary(weights)
            nonzero = quantized != 0
            scale = np.mean(np.abs(weights[nonzero])) if np.any(nonzero) else 1.0
            
        elif precision == PrecisionLevel.INT4:
            max_val = np.max(np.abs(weights))
            scale = max_val / 7.0 if max_val > 0 else 1.0
            quantized = self._quantize_int4(weights)
            
        elif precision == PrecisionLevel.INT8:
            max_val = np.max(np.abs(weights))
            scale = max_val / 127.0 if max_val > 0 else 1.0
            quantized = self._quantize_int8(weights)
            
        elif precision == PrecisionLevel.FP16:
            quantized = weights.astype(np.float16)
            scale = 1.0
            
        else:  # FP32
            quantized = weights
            scale = 1.0
        
        return quantized, scale
    
    def assign_precision(self, model: nn.Module,
                         sensitivity_scores: Optional[Dict[str, float]] = None) -> Dict[str, LayerPrecisionAssignment]:
        """
        Assign precision levels to all layers in the model.
        
        Args:
            model: Model to assign precision to
            sensitivity_scores: Optional per-layer sensitivity scores
            
        Returns:
            Dictionary of precision assignments
        """
        logger.info("Assigning precision levels to layers...")
        
        # Collect all layer names and their importance scores
        layer_scores = {}
        all_names = [name for name, _ in model.named_parameters()]
        
        for name, param in model.named_parameters():
            # Skip non-weight parameters
            if 'weight' not in name.lower() or param.ndim < 2:
                continue
            
            # Skip very small layers
            if param.numel() < 64:
                continue
            
            # Skip normalization layers
            if any(x in name.lower() for x in ['layernorm', 'layer_norm', 'rmsnorm']):
                continue
            
            sensitivity = sensitivity_scores.get(name) if sensitivity_scores else None
            score, reason = self.scorer.score_layer(name, param, all_names, sensitivity)
            layer_scores[name] = (param, score, reason)
        
        # Sort by importance score (descending)
        sorted_layers = sorted(layer_scores.items(), key=lambda x: -x[1][1])
        
        # Determine how many layers get high precision
        num_high_precision = int(len(sorted_layers) * self.config.high_precision_ratio)
        high_precision_names = set()
        
        # Add forced high-precision layers
        for pattern in self.config.force_high_precision:
            for name, _ in sorted_layers:
                if pattern in name:
                    high_precision_names.add(name)
        
        # Add top-scoring layers up to the limit
        for name, (param, score, reason) in sorted_layers:
            if len(high_precision_names) >= num_high_precision:
                break
            
            # Skip if forced to ternary
            if any(p in name for p in self.config.force_ternary):
                continue
            
            # Only consider layers above minimum size
            if param.numel() >= self.config.min_layer_size:
                high_precision_names.add(name)
        
        # Create assignments
        for name, (param, score, reason) in sorted_layers:
            if name in high_precision_names:
                precision = self.config.high_precision_level
                reason = f"high precision: {reason}"
            else:
                precision = self.config.default_precision
                reason = f"default: {reason}"
            
            memory = int(param.numel() * self._bytes_per_param(precision))
            
            self.assignments[name] = LayerPrecisionAssignment(
                name=name,
                shape=tuple(param.shape),
                numel=param.numel(),
                precision=precision,
                importance_score=score,
                reason=reason,
                memory_bytes=memory
            )
        
        logger.info(f"Assigned precision to {len(self.assignments)} layers")
        return self.assignments
    
    def get_summary(self) -> MixedPrecisionSummary:
        """Get summary of precision assignments."""
        if not self.assignments:
            return MixedPrecisionSummary(
                total_layers=0,
                precision_counts={},
                precision_params={},
                total_memory_bytes=0,
                memory_vs_fp32_ratio=1.0,
                high_precision_layers=[]
            )
        
        precision_counts = defaultdict(int)
        precision_params = defaultdict(int)
        total_memory = 0
        fp32_memory = 0
        high_precision_layers = []
        
        for assignment in self.assignments.values():
            precision_counts[assignment.precision.value] += 1
            precision_params[assignment.precision.value] += assignment.numel
            total_memory += assignment.memory_bytes
            fp32_memory += assignment.numel * 4  # FP32 = 4 bytes
            
            if assignment.precision != PrecisionLevel.TERNARY:
                high_precision_layers.append(assignment.name)
        
        return MixedPrecisionSummary(
            total_layers=len(self.assignments),
            precision_counts=dict(precision_counts),
            precision_params=dict(precision_params),
            total_memory_bytes=total_memory,
            memory_vs_fp32_ratio=total_memory / fp32_memory if fp32_memory > 0 else 1.0,
            high_precision_layers=high_precision_layers
        )
    
    def print_summary(self) -> None:
        """Print summary of precision assignments."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("MIXED-PRECISION ASSIGNMENT SUMMARY")
        print("=" * 70)
        
        print(f"\nTotal layers: {summary.total_layers}")
        
        print("\nPrecision distribution:")
        for precision, count in sorted(summary.precision_counts.items()):
            params = summary.precision_params.get(precision, 0)
            pct = count / summary.total_layers * 100
            print(f"  {precision:8s}: {count:4d} layers ({pct:5.1f}%), {params:>12,} params")
        
        print(f"\nMemory usage:")
        print(f"  Total: {summary.total_memory_bytes / 1e6:.1f} MB")
        print(f"  vs FP32: {summary.memory_vs_fp32_ratio:.1%}")
        print(f"  Compression: {1 / summary.memory_vs_fp32_ratio:.1f}x")
        
        if summary.high_precision_layers:
            print(f"\nHigh-precision layers ({len(summary.high_precision_layers)}):")
            for name in summary.high_precision_layers[:10]:
                assignment = self.assignments.get(name)
                if assignment:
                    print(f"  - {name}: {assignment.precision.value}")
            if len(summary.high_precision_layers) > 10:
                print(f"  ... and {len(summary.high_precision_layers) - 10} more")
    
    def export_assignments(self, output_path: str) -> None:
        """Export precision assignments to JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'config': {
                'default_precision': self.config.default_precision.value,
                'high_precision_ratio': self.config.high_precision_ratio,
                'high_precision_level': self.config.high_precision_level.value,
            },
            'summary': self.get_summary().to_dict(),
            'assignments': {name: a.to_dict() for name, a in self.assignments.items()}
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Assignments exported to: {output_file}")
    
    def load_assignments(self, input_path: str) -> None:
        """Load precision assignments from JSON."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        for name, assignment_data in data.get('assignments', {}).items():
            self.assignments[name] = LayerPrecisionAssignment(
                name=assignment_data['name'],
                shape=tuple(assignment_data['shape']),
                numel=assignment_data['numel'],
                precision=PrecisionLevel(assignment_data['precision']),
                importance_score=assignment_data['importance_score'],
                reason=assignment_data['reason'],
                memory_bytes=assignment_data['memory_bytes']
            )
        
        logger.info(f"Loaded assignments for {len(self.assignments)} layers")


class MixedPrecisionModelQuantizer:
    """
    High-level interface for mixed-precision model quantization.
    """
    
    def __init__(self, model_path: str,
                 config: Optional[MixedPrecisionConfig] = None,
                 device: str = 'cpu'):
        """
        Initialize the mixed-precision model quantizer.
        
        Args:
            model_path: Path to model or HuggingFace ID
            config: Mixed-precision configuration
            device: Device to use
        """
        self.model_path = model_path
        self.config = config or MixedPrecisionConfig()
        self.device = device
        self.model = None
        self.quantizer = MixedPrecisionQuantizer(self.config)
        
    def load_model(self) -> None:
        """Load the model."""
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            logger.error("transformers not installed")
            sys.exit(1)
        
        logger.info(f"Loading model: {self.model_path}")
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        
        logger.info("Model loaded successfully")
    
    def quantize(self, sensitivity_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Perform mixed-precision quantization.
        
        Args:
            sensitivity_scores: Optional per-layer sensitivity scores
            
        Returns:
            Dictionary with quantization results
        """
        if self.model is None:
            self.load_model()
        
        # Assign precision levels
        assignments = self.quantizer.assign_precision(self.model, sensitivity_scores)
        
        # Quantize each layer according to its assignment
        quantized_weights = {}
        scales = {}
        
        for name, assignment in assignments.items():
            # Get weights
            param = dict(self.model.named_parameters())[name]
            weights = param.detach().float().cpu().numpy()
            
            # Quantize
            quantized, scale = self.quantizer.quantize_layer(weights, assignment.precision)
            quantized_weights[name] = quantized
            scales[name] = scale
        
        return {
            'assignments': assignments,
            'quantized_weights': quantized_weights,
            'scales': scales,
            'summary': self.quantizer.get_summary(),
        }
    
    def export(self, output_dir: str, include_weights: bool = False) -> None:
        """
        Export quantization results.
        
        Args:
            output_dir: Output directory
            include_weights: Whether to save quantized weights
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export assignments
        self.quantizer.export_assignments(output_path / 'precision_assignments.json')
        
        logger.info(f"Exported to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mixed-precision ternary quantization for SmolVLM-256M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic mixed-precision quantization
    python mixed_precision.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    
    # Custom high-precision ratio
    python mixed_precision.py --model ./model/smolvlm-256m --high-precision-ratio 0.15
    
    # Force specific layers to high precision
    python mixed_precision.py --model HuggingFaceTB/SmolVLM-256M-Instruct \\
        --force-high-precision "lm_head,embed"
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--high-precision-ratio",
        type=float,
        default=0.1,
        help="Fraction of layers to keep at high precision"
    )
    parser.add_argument(
        "--high-precision-level",
        type=str,
        choices=['int4', 'int8', 'fp16'],
        default='int8',
        help="Precision level for high-precision layers"
    )
    parser.add_argument(
        "--force-high-precision",
        type=str,
        default=None,
        help="Comma-separated layer patterns to force to high precision"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SiLens Mixed-Precision Quantizer")
    print("=" * 70)
    
    # Build configuration
    precision_map = {
        'int4': PrecisionLevel.INT4,
        'int8': PrecisionLevel.INT8,
        'fp16': PrecisionLevel.FP16,
    }
    
    config = MixedPrecisionConfig(
        high_precision_ratio=args.high_precision_ratio,
        high_precision_level=precision_map[args.high_precision_level],
    )
    
    if args.force_high_precision:
        config.force_high_precision = args.force_high_precision.split(',')
    
    # Create quantizer and run
    quantizer = MixedPrecisionModelQuantizer(args.model, config, args.device)
    results = quantizer.quantize()
    
    # Print summary
    quantizer.quantizer.print_summary()
    
    # Export if requested
    if args.output:
        quantizer.export(args.output)
    
    print("\n" + "=" * 70)
    print("Mixed-precision quantization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
