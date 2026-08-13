#!/usr/bin/env python3
"""
Weight distribution visualizer for SmolVLM-256M.

This module provides comprehensive visualization of weight distributions,
quantization effects, and per-layer analysis using matplotlib.

Features:
- Weight distribution histograms
- Per-component heatmaps
- Quantization threshold visualization
- Before/after comparison plots
- Layer-by-layer analysis

Usage:
    python weight_visualizer.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    python weight_visualizer.py --model ./model/smolvlm-256m --output ./plots

Author: SiLens AI/ML Team
License: Apache 2.0
"""

import argparse
import logging
import sys
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


class WeightVisualizer:
    """
    Visualizes weight distributions and quantization effects.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the weight visualizer.
        
        Args:
            model_path: Path to model or HuggingFace ID
            device: Device to use
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.weights: Dict[str, np.ndarray] = {}
        
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
            self.weights[name] = param.detach().float().cpu().numpy()
        
        logger.info(f"Loaded {len(self.weights)} weight tensors")
    
    def _classify_component(self, name: str) -> str:
        """Classify weight into model component."""
        name_lower = name.lower()
        if 'vision' in name_lower or 'image' in name_lower:
            return 'vision_encoder'
        elif 'projector' in name_lower or 'connector' in name_lower:
            return 'projector'
        elif 'language' in name_lower or 'model.layers' in name_lower:
            return 'language_model'
        elif 'embed' in name_lower:
            return 'embeddings'
        elif 'lm_head' in name_lower:
            return 'lm_head'
        else:
            return 'other'
    
    def plot_overall_distribution(self, output_path: Optional[str] = None,
                                   max_samples: int = 1_000_000) -> None:
        """
        Plot overall weight distribution.
        
        Args:
            output_path: Path to save figure
            max_samples: Maximum samples for histogram
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return
        
        if not self.weights:
            self.load_model()
        
        # Collect all weights
        all_weights = []
        for name, weights in self.weights.items():
            if 'weight' in name.lower():
                all_weights.append(weights.flatten())
        
        all_weights = np.concatenate(all_weights)
        
        # Subsample if needed
        if len(all_weights) > max_samples:
            indices = np.random.choice(len(all_weights), max_samples, replace=False)
            all_weights = all_weights[indices]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Overall histogram
        axes[0, 0].hist(all_weights, bins=100, density=True, color='steelblue', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Weight Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Overall Weight Distribution')
        
        # Add statistics
        stats_text = f'μ={np.mean(all_weights):.4f}\nσ={np.std(all_weights):.4f}'
        axes[0, 0].text(0.95, 0.95, stats_text, transform=axes[0, 0].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Absolute value distribution (log scale)
        abs_weights = np.abs(all_weights)
        axes[0, 1].hist(abs_weights, bins=100, density=True, color='coral', alpha=0.7)
        axes[0, 1].set_xlabel('|Weight Value|')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Absolute Weight Distribution')
        axes[0, 1].set_yscale('log')
        
        # Add threshold lines for different alpha values
        abs_mean = np.mean(abs_weights)
        for alpha, color in [(0.5, 'green'), (0.7, 'orange'), (0.9, 'red')]:
            threshold = alpha * abs_mean
            axes[0, 1].axvline(threshold, color=color, linestyle='--', 
                              label=f'α={alpha} (τ={threshold:.3f})')
        axes[0, 1].legend()
        
        # 3. QQ plot (check for normality)
        from scipy import stats
        sorted_weights = np.sort(all_weights[:10000])  # Use subset
        theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, len(sorted_weights)))
        axes[1, 0].scatter(theoretical, sorted_weights, alpha=0.3, s=1)
        axes[1, 0].plot([-4, 4], [-4*np.std(all_weights), 4*np.std(all_weights)], 
                        'r--', alpha=0.5)
        axes[1, 0].set_xlabel('Theoretical Quantiles')
        axes[1, 0].set_ylabel('Sample Quantiles')
        axes[1, 0].set_title('Q-Q Plot (vs Normal)')
        
        # 4. Cumulative distribution
        sorted_all = np.sort(all_weights)
        cdf = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
        # Subsample for plotting
        step = max(1, len(sorted_all) // 1000)
        axes[1, 1].plot(sorted_all[::step], cdf[::step], color='purple')
        axes[1, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Weight Value')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution Function')
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_by_component(self, output_path: Optional[str] = None) -> None:
        """
        Plot weight distributions by model component.
        
        Args:
            output_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return
        
        if not self.weights:
            self.load_model()
        
        # Group by component
        by_component = defaultdict(list)
        for name, weights in self.weights.items():
            if 'weight' in name.lower() and weights.ndim >= 2:
                component = self._classify_component(name)
                by_component[component].append(weights.flatten())
        
        components = sorted(by_component.keys())
        n_components = len(components)
        
        fig, axes = plt.subplots(2, (n_components + 1) // 2, figsize=(16, 10))
        axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_components))
        
        for i, (component, color) in enumerate(zip(components, colors)):
            weights = np.concatenate(by_component[component])
            
            # Subsample
            if len(weights) > 100000:
                weights = np.random.choice(weights, 100000, replace=False)
            
            axes[i].hist(weights, bins=50, density=True, color=color, alpha=0.7)
            axes[i].axvline(0, color='red', linestyle='--', alpha=0.5)
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{component}\n(n={len(by_component[component])} layers)')
            
            # Statistics
            stats_text = f'μ={np.mean(weights):.4f}\nσ={np.std(weights):.4f}'
            axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused axes
        for i in range(n_components, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Weight Distribution by Component', fontsize=14)
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_layer_heatmap(self, output_path: Optional[str] = None,
                           max_layers: int = 50) -> None:
        """
        Plot heatmap of layer statistics.
        
        Args:
            output_path: Path to save figure
            max_layers: Maximum number of layers to show
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib/seaborn not installed")
            return
        
        if not self.weights:
            self.load_model()
        
        # Collect statistics per layer
        layer_stats = []
        layer_names = []
        
        for name, weights in self.weights.items():
            if 'weight' in name.lower() and weights.ndim >= 2:
                flat = weights.flatten()
                layer_stats.append({
                    'mean': np.mean(flat),
                    'std': np.std(flat),
                    'abs_mean': np.mean(np.abs(flat)),
                    'sparsity': np.mean(np.abs(flat) < 0.01),
                    'kurtosis': float(np.mean(((flat - np.mean(flat)) / (np.std(flat) + 1e-10)) ** 4) - 3),
                })
                layer_names.append(name.split('.')[-2] + '.' + name.split('.')[-1])
        
        # Limit layers
        if len(layer_stats) > max_layers:
            step = len(layer_stats) // max_layers
            layer_stats = layer_stats[::step][:max_layers]
            layer_names = layer_names[::step][:max_layers]
        
        # Create heatmap data
        metrics = ['mean', 'std', 'abs_mean', 'sparsity', 'kurtosis']
        data = np.array([[s[m] for m in metrics] for s in layer_stats])
        
        # Normalize each column for visualization
        data_normalized = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(10, max(8, len(layer_names) * 0.3)))
        
        sns.heatmap(data_normalized, 
                    xticklabels=metrics,
                    yticklabels=layer_names,
                    cmap='RdYlBu_r',
                    center=0,
                    ax=ax)
        
        ax.set_title('Layer Statistics Heatmap (normalized)')
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_quantization_effect(self, alpha: float = 0.7,
                                  output_path: Optional[str] = None) -> None:
        """
        Visualize quantization effect with given alpha.
        
        Args:
            alpha: Quantization threshold factor
            output_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return
        
        if not self.weights:
            self.load_model()
        
        # Pick a representative layer
        target_layer = None
        for name, weights in self.weights.items():
            if 'weight' in name.lower() and weights.ndim >= 2 and weights.size > 10000:
                target_layer = (name, weights)
                break
        
        if target_layer is None:
            logger.warning("No suitable layer found")
            return
        
        name, weights = target_layer
        flat = weights.flatten()
        
        # Quantize
        threshold = alpha * np.mean(np.abs(flat))
        quantized = np.zeros_like(flat, dtype=np.int8)
        quantized[flat > threshold] = 1
        quantized[flat < -threshold] = -1
        
        # Scale
        nonzero = quantized != 0
        scale = np.mean(np.abs(flat[nonzero])) if np.any(nonzero) else 1.0
        dequantized = quantized.astype(np.float32) * scale
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Original vs dequantized histogram
        axes[0, 0].hist(flat, bins=100, density=True, alpha=0.5, 
                        color='blue', label='Original')
        axes[0, 0].hist(dequantized, bins=100, density=True, alpha=0.5,
                        color='red', label='Dequantized')
        axes[0, 0].axvline(threshold, color='green', linestyle='--', label=f'τ={threshold:.4f}')
        axes[0, 0].axvline(-threshold, color='green', linestyle='--')
        axes[0, 0].set_xlabel('Weight Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title(f'Original vs Dequantized (α={alpha})')
        axes[0, 0].legend()
        
        # 2. Quantization error distribution
        error = flat - dequantized
        axes[0, 1].hist(error, bins=100, density=True, color='coral', alpha=0.7)
        axes[0, 1].axvline(0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Quantization Error')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title(f'Error Distribution (MAE={np.mean(np.abs(error)):.4f})')
        
        # 3. Scatter plot: original vs dequantized
        sample_idx = np.random.choice(len(flat), min(5000, len(flat)), replace=False)
        axes[1, 0].scatter(flat[sample_idx], dequantized[sample_idx], 
                           alpha=0.2, s=1)
        axes[1, 0].plot([flat.min(), flat.max()], [flat.min(), flat.max()],
                        'r--', alpha=0.5, label='y=x')
        axes[1, 0].set_xlabel('Original Weight')
        axes[1, 0].set_ylabel('Dequantized Weight')
        axes[1, 0].set_title('Original vs Dequantized')
        axes[1, 0].legend()
        
        # 4. Ternary value distribution
        counts = [np.sum(quantized == v) for v in [-1, 0, 1]]
        pcts = [c / len(quantized) * 100 for c in counts]
        bars = axes[1, 1].bar(['-1', '0', '+1'], pcts, color=['red', 'gray', 'green'])
        axes[1, 1].set_xlabel('Ternary Value')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].set_title('Ternary Distribution')
        
        for bar, pct in zip(bars, pcts):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.suptitle(f'Quantization Effect on Layer: {name}', fontsize=12)
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_layer_comparison(self, layer_names: Optional[List[str]] = None,
                              output_path: Optional[str] = None) -> None:
        """
        Plot comparison of multiple specific layers.
        
        Args:
            layer_names: Layer names to compare (uses heuristics if None)
            output_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return
        
        if not self.weights:
            self.load_model()
        
        # Select representative layers if not specified
        if layer_names is None:
            layer_names = []
            targets = ['embed', 'q_proj', 'v_proj', 'o_proj', 'mlp', 'lm_head']
            for target in targets:
                for name in self.weights:
                    if target in name.lower() and name not in layer_names:
                        layer_names.append(name)
                        break
        
        # Limit to 6 layers
        layer_names = layer_names[:6]
        n_layers = len(layer_names)
        
        if n_layers == 0:
            logger.warning("No layers found to compare")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_layers))
        
        for i, (name, color) in enumerate(zip(layer_names, colors)):
            if name not in self.weights:
                continue
            
            weights = self.weights[name].flatten()
            
            # Subsample
            if len(weights) > 50000:
                weights = np.random.choice(weights, 50000, replace=False)
            
            axes[i].hist(weights, bins=50, density=True, color=color, alpha=0.7)
            axes[i].axvline(0, color='red', linestyle='--', alpha=0.5)
            
            # Short name for title
            short_name = '.'.join(name.split('.')[-3:])
            axes[i].set_title(short_name, fontsize=10)
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Density')
            
            # Statistics annotation
            stats = f'μ={np.mean(weights):.3f}\nσ={np.std(weights):.3f}'
            axes[i].text(0.95, 0.95, stats, transform=axes[i].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused axes
        for i in range(n_layers, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Layer Weight Comparison', fontsize=14)
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_all_plots(self, output_dir: str, alpha: float = 0.7) -> None:
        """
        Generate all visualization plots.
        
        Args:
            output_dir: Directory to save plots
            alpha: Alpha for quantization visualization
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating all plots in: {output_path}")
        
        self.plot_overall_distribution(output_path / 'overall_distribution.png')
        self.plot_by_component(output_path / 'by_component.png')
        self.plot_layer_heatmap(output_path / 'layer_heatmap.png')
        self.plot_quantization_effect(alpha, output_path / 'quantization_effect.png')
        self.plot_layer_comparison(output_path=output_path / 'layer_comparison.png')
        
        logger.info("All plots generated!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Weight distribution visualizer for SmolVLM-256M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all plots
    python weight_visualizer.py --model HuggingFaceTB/SmolVLM-256M-Instruct --output ./plots
    
    # Single plot type
    python weight_visualizer.py --model ./model/smolvlm-256m --plot overall
    
    # Quantization visualization with custom alpha
    python weight_visualizer.py --model HuggingFaceTB/SmolVLM-256M-Instruct \\
        --plot quantization --alpha 0.6
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
        help="Output directory for plots"
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=['all', 'overall', 'component', 'heatmap', 'quantization', 'layers'],
        default='all',
        help="Type of plot to generate"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Alpha for quantization visualization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SiLens Weight Visualizer")
    print("=" * 70)
    
    visualizer = WeightVisualizer(args.model, device=args.device)
    
    if args.plot == 'all':
        output_dir = args.output or './plots'
        visualizer.generate_all_plots(output_dir, args.alpha)
    else:
        output_file = args.output
        if args.plot == 'overall':
            visualizer.plot_overall_distribution(output_file)
        elif args.plot == 'component':
            visualizer.plot_by_component(output_file)
        elif args.plot == 'heatmap':
            visualizer.plot_layer_heatmap(output_file)
        elif args.plot == 'quantization':
            visualizer.plot_quantization_effect(args.alpha, output_file)
        elif args.plot == 'layers':
            visualizer.plot_layer_comparison(output_path=output_file)
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
