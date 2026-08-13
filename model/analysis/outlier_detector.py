#!/usr/bin/env python3
"""
Outlier weight detector for SmolVLM-256M.

This module identifies and handles outlier weights that can significantly
impact quantization quality. Outliers are weights with unusually large
magnitudes that distort quantization thresholds and scales.

Features:
- Statistical outlier detection (z-score, IQR, MAD)
- Per-layer outlier analysis
- Outlier impact assessment on quantization
- Automatic outlier handling strategies
- Visualization of outlier distributions

Theory:
    Outliers in weights (|w| >> mean) cause:
    1. Higher quantization thresholds → more zeros
    2. Larger scale factors → lower precision for normal weights
    3. Increased reconstruction error
    
    Handling strategies:
    - Clipping: w' = clip(w, -k*σ, k*σ)
    - Percentile clipping: w' = clip(w, p_low, p_high)
    - Per-channel handling: different treatment per row/column

Usage:
    python outlier_detector.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    python outlier_detector.py --model ./model/smolvlm-256m --method iqr

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
from enum import Enum
from collections import defaultdict

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OutlierMethod(Enum):
    """Methods for outlier detection."""
    ZSCORE = "zscore"           # Z-score based (|z| > threshold)
    IQR = "iqr"                 # Interquartile range based
    MAD = "mad"                 # Median absolute deviation
    PERCENTILE = "percentile"   # Simple percentile clipping


@dataclass
class LayerOutlierAnalysis:
    """Outlier analysis for a single layer."""
    name: str
    shape: Tuple[int, ...]
    numel: int
    
    # Outlier statistics
    num_outliers: int
    outlier_fraction: float
    outlier_threshold: float
    
    # Outlier values
    max_outlier_magnitude: float
    mean_outlier_magnitude: float
    outlier_to_mean_ratio: float              # How much larger outliers are
    
    # Distribution statistics
    mean: float
    std: float
    kurtosis: float
    
    # Quantization impact
    impact_score: float                       # How much outliers affect quantization
    recommended_handling: str                 # 'none', 'clip', 'separate'
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['shape'] = list(d['shape'])
        return d


@dataclass
class OutlierSummary:
    """Summary of outlier analysis."""
    model_name: str
    total_layers: int
    method: str
    
    # Overall statistics
    total_outliers: int
    overall_outlier_fraction: float
    layers_with_significant_outliers: int
    
    # Impact assessment
    avg_impact_score: float
    high_impact_layers: List[str]
    
    # Recommendations
    layers_needing_handling: int
    recommended_clipping_percentile: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OutlierDetector:
    """
    Detects and analyzes outliers in neural network weights.
    """
    
    def __init__(self, model_path: str,
                 method: OutlierMethod = OutlierMethod.ZSCORE,
                 threshold: float = 3.0,
                 device: str = 'cpu'):
        """
        Initialize the outlier detector.
        
        Args:
            model_path: Path to model or HuggingFace ID
            method: Outlier detection method
            threshold: Detection threshold
            device: Device to use
        """
        self.model_path = model_path
        self.method = method
        self.threshold = threshold
        self.device = device
        self.model = None
        self.weights: Dict[str, np.ndarray] = {}
        self.results: Dict[str, LayerOutlierAnalysis] = {}
        
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
    
    def _detect_zscore_outliers(self, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect outliers using z-score method.
        
        Returns:
            Tuple of (outlier_mask, threshold_value)
        """
        flat = weights.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        
        if std < 1e-10:
            return np.zeros_like(flat, dtype=bool), 0.0
        
        z_scores = np.abs((flat - mean) / std)
        outlier_mask = z_scores > self.threshold
        threshold_value = self.threshold * std
        
        return outlier_mask, threshold_value
    
    def _detect_iqr_outliers(self, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect outliers using IQR method.
        
        Returns:
            Tuple of (outlier_mask, threshold_range)
        """
        flat = weights.flatten()
        q1 = np.percentile(flat, 25)
        q3 = np.percentile(flat, 75)
        iqr = q3 - q1
        
        if iqr < 1e-10:
            return np.zeros_like(flat, dtype=bool), 0.0
        
        lower = q1 - self.threshold * iqr
        upper = q3 + self.threshold * iqr
        
        outlier_mask = (flat < lower) | (flat > upper)
        threshold_value = self.threshold * iqr
        
        return outlier_mask, threshold_value
    
    def _detect_mad_outliers(self, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect outliers using Median Absolute Deviation.
        
        Returns:
            Tuple of (outlier_mask, threshold_value)
        """
        flat = weights.flatten()
        median = np.median(flat)
        mad = np.median(np.abs(flat - median))
        
        if mad < 1e-10:
            return np.zeros_like(flat, dtype=bool), 0.0
        
        # Modified z-score
        modified_z = 0.6745 * (flat - median) / mad
        outlier_mask = np.abs(modified_z) > self.threshold
        threshold_value = self.threshold * mad / 0.6745
        
        return outlier_mask, threshold_value
    
    def _detect_percentile_outliers(self, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect outliers using percentile method.
        
        threshold is interpreted as the percentile (e.g., 99.5 for top/bottom 0.5%)
        
        Returns:
            Tuple of (outlier_mask, threshold_value)
        """
        flat = weights.flatten()
        
        # Convert threshold to percentile (threshold=3 → 99.7%)
        percentile = min(99.9, 100 - (100 / self.threshold))
        
        lower = np.percentile(flat, 100 - percentile)
        upper = np.percentile(flat, percentile)
        
        outlier_mask = (flat < lower) | (flat > upper)
        threshold_value = max(abs(lower), abs(upper))
        
        return outlier_mask, threshold_value
    
    def detect_outliers(self, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect outliers using configured method.
        
        Returns:
            Tuple of (outlier_mask, threshold_value)
        """
        if self.method == OutlierMethod.ZSCORE:
            return self._detect_zscore_outliers(weights)
        elif self.method == OutlierMethod.IQR:
            return self._detect_iqr_outliers(weights)
        elif self.method == OutlierMethod.MAD:
            return self._detect_mad_outliers(weights)
        elif self.method == OutlierMethod.PERCENTILE:
            return self._detect_percentile_outliers(weights)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_kurtosis(self, weights: np.ndarray) -> float:
        """Compute excess kurtosis."""
        flat = weights.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        
        if std < 1e-10:
            return 0.0
        
        return float(np.mean(((flat - mean) / std) ** 4) - 3)
    
    def _compute_impact_score(self, weights: np.ndarray,
                               outlier_mask: np.ndarray) -> float:
        """
        Compute impact score of outliers on quantization.
        
        Score measures how much outliers distort the quantization threshold.
        """
        flat = weights.flatten()
        
        if not np.any(outlier_mask):
            return 0.0
        
        # Mean without outliers
        mean_without = np.mean(np.abs(flat[~outlier_mask]))
        
        # Mean with outliers
        mean_with = np.mean(np.abs(flat))
        
        # Impact: how much outliers shift the mean
        if mean_without > 1e-10:
            impact = (mean_with - mean_without) / mean_without
        else:
            impact = 0.0
        
        return float(impact)
    
    def _recommend_handling(self, outlier_fraction: float,
                            impact_score: float,
                            kurtosis: float) -> str:
        """Recommend outlier handling strategy."""
        
        if outlier_fraction < 0.001 and impact_score < 0.05:
            return 'none'
        elif outlier_fraction < 0.01 and impact_score < 0.15:
            return 'clip'
        elif kurtosis > 20 or impact_score > 0.3:
            return 'separate'  # Handle outliers separately
        else:
            return 'clip'
    
    def analyze_layer(self, name: str, weights: np.ndarray) -> LayerOutlierAnalysis:
        """
        Analyze outliers in a single layer.
        
        Args:
            name: Layer name
            weights: Weight tensor
            
        Returns:
            LayerOutlierAnalysis with results
        """
        flat = weights.flatten()
        
        # Detect outliers
        outlier_mask, threshold = self.detect_outliers(weights)
        
        num_outliers = int(np.sum(outlier_mask))
        outlier_fraction = num_outliers / len(flat)
        
        # Outlier statistics
        if num_outliers > 0:
            outlier_values = flat[outlier_mask]
            max_outlier = float(np.max(np.abs(outlier_values)))
            mean_outlier = float(np.mean(np.abs(outlier_values)))
        else:
            max_outlier = 0.0
            mean_outlier = 0.0
        
        # Distribution statistics
        mean = float(np.mean(flat))
        std = float(np.std(flat))
        kurtosis = self._compute_kurtosis(weights)
        
        # Outlier to mean ratio
        abs_mean = np.mean(np.abs(flat))
        if abs_mean > 1e-10:
            outlier_ratio = max_outlier / abs_mean
        else:
            outlier_ratio = 0.0
        
        # Impact score
        impact = self._compute_impact_score(weights, outlier_mask)
        
        # Recommendation
        handling = self._recommend_handling(outlier_fraction, impact, kurtosis)
        
        result = LayerOutlierAnalysis(
            name=name,
            shape=tuple(weights.shape),
            numel=weights.size,
            num_outliers=num_outliers,
            outlier_fraction=outlier_fraction,
            outlier_threshold=threshold,
            max_outlier_magnitude=max_outlier,
            mean_outlier_magnitude=mean_outlier,
            outlier_to_mean_ratio=outlier_ratio,
            mean=mean,
            std=std,
            kurtosis=kurtosis,
            impact_score=impact,
            recommended_handling=handling
        )
        
        self.results[name] = result
        return result
    
    def analyze_all_layers(self, progress: bool = True) -> Dict[str, LayerOutlierAnalysis]:
        """
        Analyze outliers in all layers.
        
        Args:
            progress: Show progress bar
            
        Returns:
            Dictionary of outlier analysis results
        """
        if not self.weights:
            self.load_model()
        
        logger.info(f"Analyzing outliers in {len(self.weights)} layers...")
        
        layers = list(self.weights.items())
        
        try:
            from tqdm import tqdm
            iterator = tqdm(layers, desc="Analyzing") if progress else layers
        except ImportError:
            iterator = layers
        
        for name, weights in iterator:
            if weights.ndim >= 2:
                self.analyze_layer(name, weights)
        
        logger.info(f"Analyzed {len(self.results)} layers")
        return self.results
    
    def get_summary(self) -> OutlierSummary:
        """Get summary of outlier analysis."""
        if not self.results:
            self.analyze_all_layers()
        
        results = list(self.results.values())
        
        # Total outliers
        total_outliers = sum(r.num_outliers for r in results)
        total_params = sum(r.numel for r in results)
        overall_fraction = total_outliers / total_params if total_params > 0 else 0.0
        
        # Significant outliers (>0.1% of layer)
        significant = sum(1 for r in results if r.outlier_fraction > 0.001)
        
        # Impact scores
        avg_impact = np.mean([r.impact_score for r in results])
        high_impact = [r.name for r in results if r.impact_score > 0.1]
        
        # Layers needing handling
        needs_handling = sum(1 for r in results if r.recommended_handling != 'none')
        
        # Recommended clipping percentile (based on average outlier fraction)
        avg_outlier_fraction = np.mean([r.outlier_fraction for r in results])
        recommended_percentile = min(99.9, 100 - avg_outlier_fraction * 100)
        
        return OutlierSummary(
            model_name=self.model_path,
            total_layers=len(results),
            method=self.method.value,
            total_outliers=total_outliers,
            overall_outlier_fraction=overall_fraction,
            layers_with_significant_outliers=significant,
            avg_impact_score=float(avg_impact),
            high_impact_layers=high_impact[:10],
            layers_needing_handling=needs_handling,
            recommended_clipping_percentile=recommended_percentile
        )
    
    def print_summary(self) -> None:
        """Print outlier analysis summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("OUTLIER ANALYSIS SUMMARY")
        print("=" * 70)
        
        print(f"\nModel: {summary.model_name}")
        print(f"Detection method: {summary.method}")
        print(f"Threshold: {self.threshold}")
        print(f"Layers analyzed: {summary.total_layers}")
        
        print(f"\n--- Outlier Statistics ---")
        print(f"Total outliers detected: {summary.total_outliers:,}")
        print(f"Overall outlier fraction: {summary.overall_outlier_fraction:.4%}")
        print(f"Layers with significant outliers (>0.1%): {summary.layers_with_significant_outliers}")
        
        print(f"\n--- Quantization Impact ---")
        print(f"Average impact score: {summary.avg_impact_score:.4f}")
        print(f"Layers needing handling: {summary.layers_needing_handling}")
        
        if summary.high_impact_layers:
            print(f"\nHigh-impact layers:")
            for name in summary.high_impact_layers[:5]:
                result = self.results.get(name)
                if result:
                    print(f"  - {name}")
                    print(f"      outliers: {result.outlier_fraction:.2%}, "
                          f"impact: {result.impact_score:.3f}, "
                          f"recommend: {result.recommended_handling}")
        
        print(f"\n--- Recommendations ---")
        print(f"Recommended clipping percentile: {summary.recommended_clipping_percentile:.1f}")
        
        if summary.avg_impact_score > 0.1:
            print("\n⚠ High outlier impact detected!")
            print("  Consider using percentile clipping before quantization:")
            print(f"  clip(weights, p_{100-summary.recommended_clipping_percentile:.1f}, "
                  f"p_{summary.recommended_clipping_percentile:.1f})")
    
    def clip_outliers(self, weights: np.ndarray,
                      percentile: float = 99.5) -> np.ndarray:
        """
        Clip outliers in weights.
        
        Args:
            weights: Original weights
            percentile: Clipping percentile
            
        Returns:
            Clipped weights
        """
        lower = np.percentile(weights, 100 - percentile)
        upper = np.percentile(weights, percentile)
        return np.clip(weights, lower, upper)
    
    def plot_outlier_distribution(self, output_path: Optional[str] = None) -> None:
        """
        Visualize outlier distribution across layers.
        
        Args:
            output_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed")
            return
        
        if not self.results:
            self.analyze_all_layers()
        
        results = list(self.results.values())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Outlier fraction distribution
        fractions = [r.outlier_fraction * 100 for r in results]
        axes[0, 0].hist(fractions, bins=50, color='coral', alpha=0.7)
        axes[0, 0].axvline(np.mean(fractions), color='red', linestyle='--',
                          label=f'Mean: {np.mean(fractions):.2f}%')
        axes[0, 0].set_xlabel('Outlier Fraction (%)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Outlier Fraction Distribution')
        axes[0, 0].legend()
        
        # 2. Impact score distribution
        impacts = [r.impact_score for r in results]
        axes[0, 1].hist(impacts, bins=50, color='steelblue', alpha=0.7)
        axes[0, 1].axvline(np.mean(impacts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(impacts):.3f}')
        axes[0, 1].set_xlabel('Impact Score')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Quantization Impact Distribution')
        axes[0, 1].legend()
        
        # 3. Kurtosis vs impact
        kurtoses = [r.kurtosis for r in results]
        axes[1, 0].scatter(kurtoses, impacts, alpha=0.5, c='purple')
        axes[1, 0].set_xlabel('Excess Kurtosis')
        axes[1, 0].set_ylabel('Impact Score')
        axes[1, 0].set_title('Kurtosis vs Impact')
        
        # 4. Outlier ratio distribution
        ratios = [r.outlier_to_mean_ratio for r in results if r.outlier_to_mean_ratio > 0]
        if ratios:
            axes[1, 1].hist(ratios, bins=50, color='forestgreen', alpha=0.7)
            axes[1, 1].axvline(np.median(ratios), color='red', linestyle='--',
                              label=f'Median: {np.median(ratios):.1f}x')
            axes[1, 1].set_xlabel('Max Outlier / Mean')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Outlier Magnitude Ratio')
            axes[1, 1].legend()
        
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
            'config': {
                'method': self.method.value,
                'threshold': self.threshold,
            },
            'summary': self.get_summary().to_dict(),
            'layers': {name: r.to_dict() for name, r in self.results.items()}
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results exported to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Outlier weight detector for SmolVLM-256M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis with z-score method
    python outlier_detector.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    
    # IQR method with custom threshold
    python outlier_detector.py --model ./model/smolvlm-256m --method iqr --threshold 1.5
    
    # With visualization and export
    python outlier_detector.py --model HuggingFaceTB/SmolVLM-256M-Instruct \\
        --plot --output ./outlier_analysis.json
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['zscore', 'iqr', 'mad', 'percentile'],
        default='zscore',
        help="Outlier detection method"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Detection threshold"
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
        help="Generate visualization"
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Path to save plot"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SiLens Outlier Detector")
    print("=" * 70)
    
    method_map = {
        'zscore': OutlierMethod.ZSCORE,
        'iqr': OutlierMethod.IQR,
        'mad': OutlierMethod.MAD,
        'percentile': OutlierMethod.PERCENTILE,
    }
    
    detector = OutlierDetector(
        args.model,
        method=method_map[args.method],
        threshold=args.threshold,
        device=args.device
    )
    
    # Run analysis
    detector.analyze_all_layers()
    
    # Print summary
    detector.print_summary()
    
    # Generate plot if requested
    if args.plot:
        detector.plot_outlier_distribution(args.plot_output)
    
    # Export if requested
    if args.output:
        detector.export_results(args.output)
    
    print("\n" + "=" * 70)
    print("Outlier analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
