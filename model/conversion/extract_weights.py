#!/usr/bin/env python3
"""
Extract and analyze all weight tensors from SmolVLM-256M.

This module provides comprehensive weight extraction and analysis functionality
for the SiLens project, preparing weights for ternary quantization and hardware
implementation.

Features:
- Complete weight extraction from all model components
- Statistical analysis (mean, std, min, max, sparsity)
- Weight distribution visualization
- Export to various formats (numpy, safetensors, JSON metadata)
- Component-level organization (vision encoder, projector, LM)

Usage:
    python extract_weights.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    python extract_weights.py --model ./model/smolvlm-256m --output ./weights
    python extract_weights.py --model HuggingFaceTB/SmolVLM-256M-Instruct --visualize

Author: SiLens AI/ML Team
License: Apache 2.0
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WeightStatistics:
    """Statistical information about a weight tensor."""
    name: str
    shape: Tuple[int, ...]
    numel: int
    dtype: str
    mean: float
    std: float
    min_val: float
    max_val: float
    abs_mean: float
    abs_median: float
    sparsity_01: float      # % of weights with |w| < 0.01
    sparsity_001: float     # % of weights with |w| < 0.001
    quantile_01: float      # 1st percentile
    quantile_99: float      # 99th percentile
    kurtosis: float         # Excess kurtosis (0 = normal)
    component: str          # vision_encoder, projector, language_model, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable types."""
        d = asdict(self)
        d['shape'] = list(d['shape'])
        return d


@dataclass
class WeightTensor:
    """Container for extracted weight tensor and its metadata."""
    name: str
    data: np.ndarray
    stats: WeightStatistics
    original_dtype: torch.dtype
    
    @property
    def component(self) -> str:
        return self.stats.component


class WeightExtractor:
    """
    Extract and analyze weights from SmolVLM-256M model.
    
    This class handles the complete weight extraction pipeline:
    1. Load model from HuggingFace or local path
    2. Extract all weight tensors
    3. Compute comprehensive statistics
    4. Organize by model component
    5. Export to various formats
    """
    
    # Component classification patterns
    COMPONENT_PATTERNS = {
        'vision_encoder': ['vision_model', 'vision_tower', 'image_encoder'],
        'projector': ['multi_modal_projector', 'connector', 'projection'],
        'language_model': ['language_model', 'model.layers', 'lm_head'],
        'embeddings': ['embed_tokens', 'wte', 'word_embeddings'],
        'normalization': ['layernorm', 'layer_norm', 'rmsnorm', 'norm'],
    }
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the weight extractor.
        
        Args:
            model_path: HuggingFace model ID or local path
            device: Device to load model on ('cpu', 'cuda', etc.)
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.weights: Dict[str, WeightTensor] = {}
        
    def load_model(self) -> None:
        """Load the model from HuggingFace or local path."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers torch")
            sys.exit(1)
        
        logger.info(f"Loading model from: {self.model_path}")
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,  # Load in FP32 for accurate analysis
            trust_remote_code=True,
            device_map=self.device
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded successfully: {total_params:,} parameters")
        
    def _classify_component(self, name: str) -> str:
        """Classify a weight tensor into its model component."""
        name_lower = name.lower()
        
        for component, patterns in self.COMPONENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return component
        
        # Default classification based on position
        if 'weight' in name_lower and 'norm' not in name_lower:
            return 'linear_weights'
        elif 'bias' in name_lower:
            return 'biases'
        
        return 'other'
    
    def _compute_statistics(self, name: str, tensor: torch.Tensor) -> WeightStatistics:
        """Compute comprehensive statistics for a weight tensor."""
        data = tensor.detach().float().cpu().numpy().flatten()
        
        # Basic statistics
        mean = float(np.mean(data))
        std = float(np.std(data))
        
        # Compute excess kurtosis (0 = normal distribution)
        if std > 1e-10:
            kurtosis = float(np.mean(((data - mean) / std) ** 4) - 3)
        else:
            kurtosis = 0.0
        
        return WeightStatistics(
            name=name,
            shape=tuple(tensor.shape),
            numel=tensor.numel(),
            dtype=str(tensor.dtype),
            mean=mean,
            std=std,
            min_val=float(np.min(data)),
            max_val=float(np.max(data)),
            abs_mean=float(np.mean(np.abs(data))),
            abs_median=float(np.median(np.abs(data))),
            sparsity_01=float(np.mean(np.abs(data) < 0.01)),
            sparsity_001=float(np.mean(np.abs(data) < 0.001)),
            quantile_01=float(np.percentile(data, 1)),
            quantile_99=float(np.percentile(data, 99)),
            kurtosis=kurtosis,
            component=self._classify_component(name)
        )
    
    def extract_all_weights(self) -> Dict[str, WeightTensor]:
        """
        Extract all weight tensors from the model.
        
        Returns:
            Dictionary mapping weight names to WeightTensor objects
        """
        if self.model is None:
            self.load_model()
        
        logger.info("Extracting weight tensors...")
        
        for name, param in self.model.named_parameters():
            stats = self._compute_statistics(name, param)
            
            self.weights[name] = WeightTensor(
                name=name,
                data=param.detach().float().cpu().numpy(),
                stats=stats,
                original_dtype=param.dtype
            )
        
        logger.info(f"Extracted {len(self.weights)} weight tensors")
        return self.weights
    
    def get_weights_by_component(self) -> Dict[str, List[WeightTensor]]:
        """Group weights by their model component."""
        by_component = defaultdict(list)
        
        for weight in self.weights.values():
            by_component[weight.component].append(weight)
        
        return dict(by_component)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all weights.
        
        Returns:
            Dictionary with aggregate statistics
        """
        if not self.weights:
            self.extract_all_weights()
        
        by_component = self.get_weights_by_component()
        
        summary = {
            'total_weights': len(self.weights),
            'total_parameters': sum(w.stats.numel for w in self.weights.values()),
            'components': {},
            'global_stats': {},
        }
        
        # Per-component statistics
        for component, weights in by_component.items():
            total_params = sum(w.stats.numel for w in weights)
            
            # Weighted averages
            all_std = [w.stats.std for w in weights]
            all_sparsity = [w.stats.sparsity_01 for w in weights]
            
            summary['components'][component] = {
                'count': len(weights),
                'total_params': total_params,
                'avg_std': float(np.mean(all_std)),
                'avg_sparsity': float(np.mean(all_sparsity)),
            }
        
        # Global statistics (parameter-weighted)
        all_abs_means = []
        all_weights_flat = []
        
        for w in self.weights.values():
            if w.stats.numel > 0:
                all_abs_means.append(w.stats.abs_mean)
                all_weights_flat.append(w.data.flatten())
        
        if all_weights_flat:
            concatenated = np.concatenate(all_weights_flat)
            summary['global_stats'] = {
                'global_mean': float(np.mean(concatenated)),
                'global_std': float(np.std(concatenated)),
                'global_abs_mean': float(np.mean(np.abs(concatenated))),
                'global_sparsity_01': float(np.mean(np.abs(concatenated) < 0.01)),
            }
        
        return summary
    
    def print_statistics(self, detailed: bool = False) -> None:
        """Print weight statistics to console."""
        if not self.weights:
            self.extract_all_weights()
        
        summary = self.get_summary_statistics()
        
        print("\n" + "=" * 70)
        print("WEIGHT EXTRACTION SUMMARY")
        print("=" * 70)
        
        print(f"\nTotal weight tensors: {summary['total_weights']}")
        print(f"Total parameters: {summary['total_parameters']:,}")
        
        print("\n--- By Component ---")
        for comp, stats in sorted(summary['components'].items(), 
                                   key=lambda x: -x[1]['total_params']):
            pct = 100 * stats['total_params'] / summary['total_parameters']
            print(f"\n{comp.upper()}")
            print(f"  Tensors: {stats['count']}")
            print(f"  Parameters: {stats['total_params']:,} ({pct:.1f}%)")
            print(f"  Avg std: {stats['avg_std']:.6f}")
            print(f"  Avg sparsity (<0.01): {stats['avg_sparsity']:.1%}")
        
        if 'global_stats' in summary and summary['global_stats']:
            print("\n--- Global Statistics ---")
            gs = summary['global_stats']
            print(f"  Global mean: {gs['global_mean']:.6f}")
            print(f"  Global std: {gs['global_std']:.6f}")
            print(f"  Global |mean|: {gs['global_abs_mean']:.6f}")
            print(f"  Global sparsity: {gs['global_sparsity_01']:.1%}")
        
        if detailed:
            print("\n--- Detailed Per-Layer Statistics ---")
            for name, weight in sorted(self.weights.items()):
                s = weight.stats
                print(f"\n{name}")
                print(f"  Shape: {s.shape}")
                print(f"  Params: {s.numel:,}")
                print(f"  Mean: {s.mean:.6f}, Std: {s.std:.6f}")
                print(f"  Range: [{s.min_val:.4f}, {s.max_val:.4f}]")
                print(f"  |Mean|: {s.abs_mean:.6f}")
                print(f"  Sparsity: {s.sparsity_01:.1%}")
                print(f"  Kurtosis: {s.kurtosis:.2f}")
    
    def export_weights(self, output_dir: str, 
                       format: str = 'numpy',
                       include_metadata: bool = True) -> None:
        """
        Export extracted weights to files.
        
        Args:
            output_dir: Directory to save weights
            format: Export format ('numpy', 'safetensors', 'both')
            include_metadata: Whether to export JSON metadata
        """
        if not self.weights:
            self.extract_all_weights()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting weights to: {output_path}")
        
        # Export by component
        by_component = self.get_weights_by_component()
        
        for component, weights in by_component.items():
            comp_dir = output_path / component
            comp_dir.mkdir(exist_ok=True)
            
            if format in ('numpy', 'both'):
                # Save as numpy files
                for w in weights:
                    # Sanitize name for filename
                    safe_name = w.name.replace('.', '_').replace('/', '_')
                    np.save(comp_dir / f"{safe_name}.npy", w.data)
            
            if format in ('safetensors', 'both'):
                # Save component as single safetensors file
                try:
                    from safetensors.numpy import save_file
                    tensors = {w.name: w.data for w in weights}
                    save_file(tensors, comp_dir / f"{component}_weights.safetensors")
                except ImportError:
                    logger.warning("safetensors not installed, skipping safetensors export")
        
        if include_metadata:
            # Export metadata
            metadata = {
                'model_path': self.model_path,
                'summary': self.get_summary_statistics(),
                'weights': {name: w.stats.to_dict() for name, w in self.weights.items()}
            }
            
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to: {output_path / 'metadata.json'}")
        
        logger.info(f"Export complete: {len(self.weights)} tensors saved")
    
    def visualize_distributions(self, output_path: Optional[str] = None) -> None:
        """
        Visualize weight distributions.
        
        Args:
            output_path: If provided, save plots to this directory
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed, skipping visualization")
            return
        
        if not self.weights:
            self.extract_all_weights()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Overall weight distribution
        all_weights = np.concatenate([w.data.flatten() for w in self.weights.values()])
        # Subsample for performance
        if len(all_weights) > 1_000_000:
            indices = np.random.choice(len(all_weights), 1_000_000, replace=False)
            all_weights = all_weights[indices]
        
        axes[0, 0].hist(all_weights, bins=100, density=True, alpha=0.7, color='steelblue')
        axes[0, 0].set_xlabel('Weight Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Overall Weight Distribution')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Distribution by component
        by_component = self.get_weights_by_component()
        component_means = []
        for comp, weights in by_component.items():
            data = np.concatenate([w.data.flatten() for w in weights])
            component_means.append({
                'component': comp,
                'abs_mean': np.mean(np.abs(data)),
                'std': np.std(data)
            })
        
        components = [c['component'] for c in component_means]
        abs_means = [c['abs_mean'] for c in component_means]
        
        axes[0, 1].barh(components, abs_means, color='steelblue')
        axes[0, 1].set_xlabel('Mean |Weight|')
        axes[0, 1].set_title('Mean Absolute Weight by Component')
        
        # 3. Sparsity by component
        sparsities = []
        for comp, weights in by_component.items():
            data = np.concatenate([w.data.flatten() for w in weights])
            sparsities.append(np.mean(np.abs(data) < 0.01) * 100)
        
        axes[1, 0].barh(components, sparsities, color='coral')
        axes[1, 0].set_xlabel('Sparsity (% < 0.01)')
        axes[1, 0].set_title('Weight Sparsity by Component')
        
        # 4. Standard deviation distribution
        stds = [w.stats.std for w in self.weights.values()]
        axes[1, 1].hist(stds, bins=50, color='forestgreen', alpha=0.7)
        axes[1, 1].set_xlabel('Standard Deviation')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Layer Std Deviations')
        
        plt.tight_layout()
        
        if output_path:
            output_file = Path(output_path) / 'weight_distributions.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def get_quantization_recommendations(self) -> Dict[str, Any]:
        """
        Analyze weights and provide quantization recommendations.
        
        Returns:
            Dictionary with recommendations for each weight tensor
        """
        if not self.weights:
            self.extract_all_weights()
        
        recommendations = {
            'easy': [],       # Standard ternary quantization
            'moderate': [],   # May need calibration
            'difficult': [],  # May need special handling
        }
        
        for name, weight in self.weights.items():
            s = weight.stats
            
            # Classification criteria
            # Easy: low kurtosis, not too sparse, symmetric
            # Difficult: high kurtosis (heavy tails), very sparse, or asymmetric
            
            is_symmetric = abs(s.mean) < 0.1 * s.std
            low_kurtosis = s.kurtosis < 5
            moderate_sparsity = s.sparsity_01 < 0.5
            
            if is_symmetric and low_kurtosis and moderate_sparsity:
                difficulty = 'easy'
            elif (is_symmetric or low_kurtosis) and s.kurtosis < 15:
                difficulty = 'moderate'
            else:
                difficulty = 'difficult'
            
            rec = {
                'name': name,
                'shape': s.shape,
                'abs_mean': s.abs_mean,
                'recommended_threshold': s.abs_mean * 0.7,  # α = 0.7
                'notes': []
            }
            
            if not is_symmetric:
                rec['notes'].append('Asymmetric distribution')
            if s.kurtosis > 10:
                rec['notes'].append(f'High kurtosis ({s.kurtosis:.1f})')
            if s.sparsity_01 > 0.3:
                rec['notes'].append(f'High sparsity ({s.sparsity_01:.1%})')
            
            recommendations[difficulty].append(rec)
        
        return recommendations


def main():
    """Main entry point for weight extraction."""
    parser = argparse.ArgumentParser(
        description="Extract and analyze weights from SmolVLM-256M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract from HuggingFace Hub
    python extract_weights.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    
    # Extract from local model and save to custom directory
    python extract_weights.py --model ./model/smolvlm-256m --output ./weights
    
    # Include visualization
    python extract_weights.py --model HuggingFaceTB/SmolVLM-256M-Instruct --visualize
    
    # Detailed statistics
    python extract_weights.py --model HuggingFaceTB/SmolVLM-256M-Instruct --detailed
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
        default="./model/weights/extracted",
        help="Output directory for extracted weights"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['numpy', 'safetensors', 'both'],
        default='numpy',
        help="Export format"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to load model on"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-layer statistics"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate weight distribution visualizations"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export weights to files"
    )
    parser.add_argument(
        "--recommendations",
        action="store_true",
        help="Print quantization recommendations"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SiLens Weight Extractor")
    print("=" * 70)
    
    # Initialize extractor
    extractor = WeightExtractor(args.model, device=args.device)
    
    # Load and extract
    extractor.load_model()
    extractor.extract_all_weights()
    
    # Print statistics
    extractor.print_statistics(detailed=args.detailed)
    
    # Export if requested
    if args.export:
        extractor.export_weights(args.output, format=args.format)
    
    # Visualize if requested
    if args.visualize:
        extractor.visualize_distributions(args.output if args.export else None)
    
    # Print recommendations if requested
    if args.recommendations:
        print("\n" + "=" * 70)
        print("QUANTIZATION RECOMMENDATIONS")
        print("=" * 70)
        
        recs = extractor.get_quantization_recommendations()
        
        for difficulty, layers in recs.items():
            print(f"\n{difficulty.upper()} ({len(layers)} layers):")
            for r in layers[:5]:  # Show first 5
                notes = ', '.join(r['notes']) if r['notes'] else 'None'
                print(f"  {r['name']}")
                print(f"    Threshold: {r['recommended_threshold']:.6f}")
                print(f"    Notes: {notes}")
            if len(layers) > 5:
                print(f"  ... and {len(layers) - 5} more")
    
    print("\n" + "=" * 70)
    print("Extraction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
