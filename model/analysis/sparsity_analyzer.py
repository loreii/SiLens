#!/usr/bin/env python3
"""
Sparsity pattern analyzer for SmolVLM-256M weights.

This module analyzes sparsity patterns in model weights, helping to:
- Understand natural sparsity in the model
- Predict quantization-induced sparsity
- Identify structured sparsity patterns
- Optimize quantization parameters

Features:
- Per-layer sparsity analysis
- Threshold-dependent sparsity curves
- Structured vs unstructured sparsity detection
- Block sparsity analysis
- Component-level sparsity comparison

Usage:
    python sparsity_analyzer.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    python sparsity_analyzer.py --model ./model/smolvlm-256m --threshold 0.01

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LayerSparsity:
    """Sparsity analysis for a single layer."""
    name: str
    shape: Tuple[int, ...]
    numel: int
    
    # Sparsity at different thresholds
    sparsity_0001: float                      # |w| < 0.001
    sparsity_001: float                       # |w| < 0.01
    sparsity_01: float                        # |w| < 0.1
    
    # Ternary quantization sparsity (predicted)
    ternary_sparsity_05: float                # α = 0.5
    ternary_sparsity_07: float                # α = 0.7
    ternary_sparsity_09: float                # α = 0.9
    
    # Structured sparsity
    row_sparsity: float                       # Fraction of near-zero rows
    column_sparsity: float                    # Fraction of near-zero columns
    block_sparsity_8: float                   # 8x8 block sparsity
    block_sparsity_16: float                  # 16x16 block sparsity
    
    # Patterns
    has_structured_pattern: bool
    pattern_type: str                         # 'none', 'row', 'column', 'block'
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['shape'] = list(d['shape'])
        return d


@dataclass
class SparsitySummary:
    """Summary of sparsity analysis."""
    model_name: str
    total_layers: int
    total_params: int
    
    # Overall sparsity
    overall_sparsity_001: float
    overall_ternary_sparsity: float
    
    # By component
    component_sparsity: Dict[str, float]
    
    # Structured patterns
    layers_with_structured_sparsity: int
    dominant_pattern: str
    
    # Recommendations
    estimated_compression: float
    recommended_alpha: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SparsityAnalyzer:
    """
    Analyzes sparsity patterns in neural network weights.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the sparsity analyzer.
        
        Args:
            model_path: Path to model or HuggingFace ID
            device: Device to use
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.weights: Dict[str, np.ndarray] = {}
        self.results: Dict[str, LayerSparsity] = {}
        
    def load_model(self) -> None:
        """Load the model."""
        try:
            import torch
            from transformers import AutoModelForVision2Seq
        except ImportError:
            logger.error("torch/transformers not installed")
            sys.exit(1)
        
        logger.info(f"Loading model: {self.model_path}")
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        
        # Extract weights
        for name, param in self.model.named_parameters():
            if 'weight' in name.lower():
                self.weights[name] = param.detach().float().cpu().numpy()
        
        logger.info(f"Loaded {len(self.weights)} weight tensors")
    
    def _classify_component(self, name: str) -> str:
        """Classify weight into model component."""
        name_lower = name.lower()
        if 'vision' in name_lower or 'image' in name_lower:
            return 'vision_encoder'
        elif 'projector' in name_lower:
            return 'projector'
        elif 'language' in name_lower or 'model.layers' in name_lower:
            return 'language_model'
        elif 'embed' in name_lower:
            return 'embeddings'
        elif 'lm_head' in name_lower:
            return 'lm_head'
        else:
            return 'other'
    
    def _compute_threshold_sparsity(self, weights: np.ndarray, 
                                     threshold: float) -> float:
        """Compute sparsity at a given threshold."""
        return float(np.mean(np.abs(weights) < threshold))
    
    def _compute_ternary_sparsity(self, weights: np.ndarray, 
                                   alpha: float) -> float:
        """Compute predicted ternary quantization sparsity."""
        abs_mean = np.mean(np.abs(weights))
        threshold = alpha * abs_mean
        return float(np.mean(np.abs(weights) <= threshold))
    
    def _compute_row_sparsity(self, weights: np.ndarray, 
                               threshold: float = 0.01) -> float:
        """Compute fraction of near-zero rows."""
        if weights.ndim < 2:
            return 0.0
        
        row_norms = np.linalg.norm(weights, axis=1)
        return float(np.mean(row_norms < threshold * weights.shape[1]))
    
    def _compute_column_sparsity(self, weights: np.ndarray,
                                  threshold: float = 0.01) -> float:
        """Compute fraction of near-zero columns."""
        if weights.ndim < 2:
            return 0.0
        
        col_norms = np.linalg.norm(weights, axis=0)
        return float(np.mean(col_norms < threshold * weights.shape[0]))
    
    def _compute_block_sparsity(self, weights: np.ndarray,
                                 block_size: int,
                                 threshold: float = 0.01) -> float:
        """Compute block sparsity."""
        if weights.ndim < 2:
            return 0.0
        
        rows, cols = weights.shape
        
        # Pad to block size
        pad_rows = (block_size - rows % block_size) % block_size
        pad_cols = (block_size - cols % block_size) % block_size
        
        if pad_rows > 0 or pad_cols > 0:
            padded = np.pad(weights, ((0, pad_rows), (0, pad_cols)), mode='constant')
        else:
            padded = weights
        
        # Compute block-wise norms
        n_blocks_rows = padded.shape[0] // block_size
        n_blocks_cols = padded.shape[1] // block_size
        
        sparse_blocks = 0
        total_blocks = n_blocks_rows * n_blocks_cols
        
        for i in range(n_blocks_rows):
            for j in range(n_blocks_cols):
                block = padded[i*block_size:(i+1)*block_size,
                              j*block_size:(j+1)*block_size]
                if np.linalg.norm(block) < threshold * block_size * block_size:
                    sparse_blocks += 1
        
        return sparse_blocks / total_blocks if total_blocks > 0 else 0.0
    
    def _detect_structured_pattern(self, weights: np.ndarray) -> Tuple[bool, str]:
        """Detect if weights have structured sparsity pattern."""
        if weights.ndim < 2:
            return False, 'none'
        
        row_sparsity = self._compute_row_sparsity(weights)
        col_sparsity = self._compute_column_sparsity(weights)
        block_sparsity = self._compute_block_sparsity(weights, 8)
        
        # Threshold for considering pattern as structured
        pattern_threshold = 0.1
        
        if row_sparsity > pattern_threshold and row_sparsity > col_sparsity:
            return True, 'row'
        elif col_sparsity > pattern_threshold and col_sparsity > row_sparsity:
            return True, 'column'
        elif block_sparsity > pattern_threshold:
            return True, 'block'
        else:
            return False, 'none'
    
    def analyze_layer(self, name: str, weights: np.ndarray) -> LayerSparsity:
        """
        Analyze sparsity for a single layer.
        
        Args:
            name: Layer name
            weights: Weight tensor
            
        Returns:
            LayerSparsity with analysis results
        """
        # Threshold-based sparsity
        sparsity_0001 = self._compute_threshold_sparsity(weights, 0.001)
        sparsity_001 = self._compute_threshold_sparsity(weights, 0.01)
        sparsity_01 = self._compute_threshold_sparsity(weights, 0.1)
        
        # Ternary sparsity predictions
        ternary_05 = self._compute_ternary_sparsity(weights, 0.5)
        ternary_07 = self._compute_ternary_sparsity(weights, 0.7)
        ternary_09 = self._compute_ternary_sparsity(weights, 0.9)
        
        # Structured sparsity
        if weights.ndim >= 2:
            row_sparsity = self._compute_row_sparsity(weights)
            col_sparsity = self._compute_column_sparsity(weights)
            block_8 = self._compute_block_sparsity(weights, 8)
            block_16 = self._compute_block_sparsity(weights, 16)
        else:
            row_sparsity = col_sparsity = block_8 = block_16 = 0.0
        
        has_pattern, pattern_type = self._detect_structured_pattern(weights)
        
        result = LayerSparsity(
            name=name,
            shape=tuple(weights.shape),
            numel=weights.size,
            sparsity_0001=sparsity_0001,
            sparsity_001=sparsity_001,
            sparsity_01=sparsity_01,
            ternary_sparsity_05=ternary_05,
            ternary_sparsity_07=ternary_07,
            ternary_sparsity_09=ternary_09,
            row_sparsity=row_sparsity,
            column_sparsity=col_sparsity,
            block_sparsity_8=block_8,
            block_sparsity_16=block_16,
            has_structured_pattern=has_pattern,
            pattern_type=pattern_type
        )
        
        self.results[name] = result
        return result
    
    def analyze_all_layers(self, progress: bool = True) -> Dict[str, LayerSparsity]:
        """
        Analyze sparsity for all layers.
        
        Args:
            progress: Show progress bar
            
        Returns:
            Dictionary of sparsity analysis results
        """
        if not self.weights:
            self.load_model()
        
        logger.info(f"Analyzing sparsity for {len(self.weights)} layers...")
        
        layers = list(self.weights.items())
        
        try:
            from tqdm import tqdm
            iterator = tqdm(layers, desc="Analyzing") if progress else layers
        except ImportError:
            iterator = layers
        
        for name, weights in iterator:
            if weights.ndim >= 2:  # Only analyze weight matrices
                self.analyze_layer(name, weights)
        
        logger.info(f"Analyzed {len(self.results)} layers")
        return self.results
    
    def get_summary(self) -> SparsitySummary:
        """Get summary of sparsity analysis."""
        if not self.results:
            self.analyze_all_layers()
        
        results = list(self.results.values())
        
        # Overall sparsity
        total_params = sum(r.numel for r in results)
        weighted_sparsity_001 = sum(r.sparsity_001 * r.numel for r in results) / total_params
        weighted_ternary = sum(r.ternary_sparsity_07 * r.numel for r in results) / total_params
        
        # By component
        component_sparsity = defaultdict(list)
        for r in results:
            component = self._classify_component(r.name)
            component_sparsity[component].append(r.ternary_sparsity_07)
        
        component_avg = {c: np.mean(s) for c, s in component_sparsity.items()}
        
        # Structured patterns
        structured_count = sum(1 for r in results if r.has_structured_pattern)
        patterns = [r.pattern_type for r in results if r.has_structured_pattern]
        if patterns:
            from collections import Counter
            dominant = Counter(patterns).most_common(1)[0][0]
        else:
            dominant = 'none'
        
        # Recommendations
        # Estimate compression: ternary uses 2 bits, with sparsity we can do better
        effective_bits = 2 * (1 - weighted_ternary) + 0.1 * weighted_ternary
        estimated_compression = 32 / effective_bits
        
        # Recommend alpha based on sparsity-accuracy tradeoff
        if weighted_ternary < 0.25:
            recommended_alpha = 0.8
        elif weighted_ternary < 0.40:
            recommended_alpha = 0.7
        else:
            recommended_alpha = 0.6
        
        return SparsitySummary(
            model_name=self.model_path,
            total_layers=len(results),
            total_params=total_params,
            overall_sparsity_001=weighted_sparsity_001,
            overall_ternary_sparsity=weighted_ternary,
            component_sparsity=component_avg,
            layers_with_structured_sparsity=structured_count,
            dominant_pattern=dominant,
            estimated_compression=estimated_compression,
            recommended_alpha=recommended_alpha
        )
    
    def print_summary(self) -> None:
        """Print sparsity analysis summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("SPARSITY ANALYSIS SUMMARY")
        print("=" * 70)
        
        print(f"\nModel: {summary.model_name}")
        print(f"Layers analyzed: {summary.total_layers}")
        print(f"Total parameters: {summary.total_params:,}")
        
        print(f"\n--- Natural Sparsity ---")
        print(f"Near-zero weights (|w| < 0.01): {summary.overall_sparsity_001:.1%}")
        
        print(f"\n--- Predicted Ternary Sparsity (α=0.7) ---")
        print(f"Overall: {summary.overall_ternary_sparsity:.1%}")
        
        print(f"\nBy component:")
        for component, sparsity in sorted(summary.component_sparsity.items()):
            bar = "█" * int(sparsity * 20)
            print(f"  {component:20s}: {sparsity:5.1%} {bar}")
        
        print(f"\n--- Structured Sparsity ---")
        print(f"Layers with patterns: {summary.layers_with_structured_sparsity}")
        print(f"Dominant pattern: {summary.dominant_pattern}")
        
        print(f"\n--- Recommendations ---")
        print(f"Estimated compression ratio: {summary.estimated_compression:.1f}x")
        print(f"Recommended alpha: {summary.recommended_alpha}")
    
    def plot_sparsity_curves(self, output_path: Optional[str] = None) -> None:
        """
        Plot sparsity vs threshold curves.
        
        Args:
            output_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed")
            return
        
        if not self.weights:
            self.load_model()
        
        # Compute sparsity at various thresholds
        thresholds = np.logspace(-4, 0, 50)
        alpha_values = np.linspace(0.3, 1.0, 30)
        
        # Aggregate over all weights
        all_weights = []
        for name, weights in self.weights.items():
            if weights.ndim >= 2:
                all_weights.append(weights.flatten())
        
        all_weights = np.concatenate(all_weights)
        abs_mean = np.mean(np.abs(all_weights))
        
        # Compute sparsities
        threshold_sparsity = [np.mean(np.abs(all_weights) < t) for t in thresholds]
        ternary_sparsity = [np.mean(np.abs(all_weights) <= a * abs_mean) for a in alpha_values]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Threshold-based sparsity
        axes[0].plot(thresholds, threshold_sparsity, 'b-', linewidth=2)
        axes[0].axhline(0.3, color='gray', linestyle='--', alpha=0.5, label='30%')
        axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Sparsity')
        axes[0].set_title('Sparsity vs Threshold')
        axes[0].set_xscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Alpha-based ternary sparsity
        axes[1].plot(alpha_values, ternary_sparsity, 'r-', linewidth=2)
        axes[1].axvline(0.7, color='green', linestyle='--', alpha=0.7, label='α=0.7')
        axes[1].axhline(0.35, color='gray', linestyle='--', alpha=0.5, label='35%')
        axes[1].set_xlabel('Alpha (α)')
        axes[1].set_ylabel('Ternary Sparsity (zeros)')
        axes[1].set_title('Ternary Sparsity vs Alpha')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_results(self, output_path: str) -> None:
        """Export analysis results to JSON."""
        if not self.results:
            self.analyze_all_layers()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'summary': self.get_summary().to_dict(),
            'layers': {name: r.to_dict() for name, r in self.results.items()}
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results exported to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sparsity pattern analyzer for SmolVLM-256M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python sparsity_analyzer.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    
    # With visualization
    python sparsity_analyzer.py --model ./model/smolvlm-256m --plot
    
    # Export results
    python sparsity_analyzer.py --model HuggingFaceTB/SmolVLM-256M-Instruct \\
        --output ./sparsity_analysis.json
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Export results to JSON"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate sparsity plots"
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Path to save plots"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SiLens Sparsity Analyzer")
    print("=" * 70)
    
    analyzer = SparsityAnalyzer(args.model, device=args.device)
    analyzer.analyze_all_layers()
    
    # Print summary
    analyzer.print_summary()
    
    # Generate plots if requested
    if args.plot:
        analyzer.plot_sparsity_curves(args.plot_output)
    
    # Export if requested
    if args.output:
        analyzer.export_results(args.output)
    
    print("\n" + "=" * 70)
    print("Sparsity analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
