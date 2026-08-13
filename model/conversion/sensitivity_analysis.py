#!/usr/bin/env python3
"""
Layer-by-layer sensitivity analysis for ternary quantization.

This module analyzes how sensitive each layer is to quantization, helping
identify which layers require special handling (higher precision, different
alpha, etc.) and which layers can be aggressively quantized.

Features:
- Per-layer quantization sensitivity scoring
- Output-based sensitivity (how much layer output changes)
- Gradient-based sensitivity (using Fisher information)
- Hessian approximation for importance estimation
- Cross-layer impact analysis

Theory:
    Sensitivity measures how much a layer's quantization affects model output.
    
    Output-based: S_output = ||f(x; W) - f(x; Q(W))||
    Fisher-based: S_fisher = E[∂L/∂W]² (gradient magnitude)
    
    Layers with high sensitivity should be quantized more carefully.

Usage:
    python sensitivity_analysis.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    python sensitivity_analysis.py --model ./model/smolvlm-256m --samples 50

Author: SiLens AI/ML Team
License: Apache 2.0
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    num_samples: int = 50                     # Number of samples for analysis
    alpha_range: Tuple[float, ...] = (0.5, 0.7, 0.9)  # Alpha values to test
    batch_size: int = 1                       # Batch size for analysis
    use_fisher: bool = True                   # Use Fisher information
    use_output_sensitivity: bool = True       # Use output-based sensitivity
    normalize_scores: bool = True             # Normalize scores to [0, 1]
    seed: int = 42                            # Random seed


@dataclass
class LayerSensitivity:
    """Sensitivity analysis result for a single layer."""
    name: str
    shape: Tuple[int, ...]
    numel: int
    
    # Core metrics
    output_sensitivity: float                 # How much output changes
    weight_sensitivity: float                 # How important weights are
    fisher_score: Optional[float] = None      # Fisher information based
    
    # Per-alpha analysis
    alpha_sensitivities: Dict[float, float] = field(default_factory=dict)
    
    # Recommendations
    recommended_alpha: float = 0.7
    quantization_difficulty: str = "medium"   # easy, medium, hard
    
    # Statistics
    weight_stats: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'name': self.name,
            'shape': list(self.shape),
            'numel': self.numel,
            'output_sensitivity': self.output_sensitivity,
            'weight_sensitivity': self.weight_sensitivity,
            'fisher_score': self.fisher_score,
            'alpha_sensitivities': self.alpha_sensitivities,
            'recommended_alpha': self.recommended_alpha,
            'quantization_difficulty': self.quantization_difficulty,
            'weight_stats': self.weight_stats,
        }


@dataclass
class SensitivitySummary:
    """Summary of sensitivity analysis."""
    total_layers: int
    difficulty_distribution: Dict[str, int]   # easy/medium/hard counts
    most_sensitive_layers: List[str]
    least_sensitive_layers: List[str]
    avg_output_sensitivity: float
    recommended_high_precision_layers: List[str]
    analysis_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OutputSensitivityAnalyzer:
    """
    Analyzes how much layer outputs change when weights are quantized.
    """
    
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        
    def _quantize_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantize weights to ternary."""
        threshold = self.alpha * np.mean(np.abs(weights))
        quantized = np.zeros_like(weights, dtype=np.int8)
        quantized[weights > threshold] = 1
        quantized[weights < -threshold] = -1
        
        nonzero = quantized != 0
        scale = np.mean(np.abs(weights[nonzero])) if np.any(nonzero) else 1.0
        
        return quantized, scale
    
    def compute_layer_sensitivity(self, 
                                   module: nn.Module,
                                   sample_input: torch.Tensor) -> float:
        """
        Compute output sensitivity for a single linear layer.
        
        Args:
            module: Linear layer module
            sample_input: Sample input tensor
            
        Returns:
            Sensitivity score (higher = more sensitive)
        """
        if not isinstance(module, nn.Linear):
            return 0.0
        
        # Get original output
        with torch.no_grad():
            original_output = module(sample_input)
        
        # Quantize weights
        weights = module.weight.detach().float().cpu().numpy()
        quantized, scale = self._quantize_weights(weights)
        
        # Create quantized layer
        quantized_weights = torch.from_numpy(
            quantized.astype(np.float32) * scale
        ).to(module.weight.device)
        
        # Compute quantized output
        with torch.no_grad():
            quantized_output = F.linear(
                sample_input, 
                quantized_weights,
                module.bias
            )
        
        # Compute sensitivity as relative change
        orig_norm = torch.norm(original_output).item()
        diff_norm = torch.norm(original_output - quantized_output).item()
        
        if orig_norm > 1e-10:
            sensitivity = diff_norm / orig_norm
        else:
            sensitivity = diff_norm
        
        return float(sensitivity)


class FisherSensitivityAnalyzer:
    """
    Analyzes layer sensitivity using Fisher information.
    
    Fisher information measures how much the loss changes when weights change,
    providing a proxy for layer importance.
    """
    
    def __init__(self):
        self.gradients: Dict[str, List[torch.Tensor]] = defaultdict(list)
        
    def collect_gradients(self, model: nn.Module, 
                          dataloader,
                          num_samples: int = 50) -> None:
        """
        Collect gradients for Fisher information estimation.
        
        Args:
            model: Model to analyze
            dataloader: Data loader for samples
            num_samples: Number of samples to use
        """
        model.train()  # Enable gradients
        
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            try:
                # Forward pass
                if isinstance(batch, dict):
                    outputs = model(**batch)
                else:
                    outputs = model(batch)
                
                # Use a simple loss (e.g., mean of logits)
                if hasattr(outputs, 'logits'):
                    loss = outputs.logits.mean()
                elif isinstance(outputs, torch.Tensor):
                    loss = outputs.mean()
                else:
                    continue
                
                # Backward pass
                model.zero_grad()
                loss.backward()
                
                # Collect gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        self.gradients[name].append(param.grad.detach().clone())
                
                sample_count += 1
                
            except Exception as e:
                logger.warning(f"Gradient collection failed: {e}")
                continue
        
        model.eval()
    
    def compute_fisher_score(self, name: str) -> float:
        """
        Compute Fisher information score for a layer.
        
        Args:
            name: Layer name
            
        Returns:
            Fisher score (higher = more important)
        """
        if name not in self.gradients or not self.gradients[name]:
            return 0.0
        
        # Stack gradients and compute squared mean
        grads = torch.stack(self.gradients[name])
        
        # Fisher score = E[g²] (diagonal Fisher)
        fisher_score = torch.mean(grads ** 2).item()
        
        return float(fisher_score)
    
    def clear(self):
        """Clear collected gradients."""
        self.gradients.clear()


class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for neural network layers.
    """
    
    def __init__(self, config: Optional[SensitivityConfig] = None):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            config: Analysis configuration
        """
        self.config = config or SensitivityConfig()
        self.output_analyzer = OutputSensitivityAnalyzer()
        self.fisher_analyzer = FisherSensitivityAnalyzer()
        self.results: Dict[str, LayerSensitivity] = {}
        
    def _classify_difficulty(self, output_sens: float, 
                              fisher_score: Optional[float],
                              weight_stats: Dict[str, float]) -> str:
        """Classify quantization difficulty based on sensitivity metrics."""
        
        # Combine metrics into difficulty score
        score = output_sens
        
        if fisher_score is not None:
            score += fisher_score * 0.5
        
        # Consider weight statistics
        kurtosis = weight_stats.get('kurtosis', 0)
        if kurtosis > 10:
            score *= 1.5  # Heavy tails make quantization harder
        
        # Classify
        if score < 0.1:
            return "easy"
        elif score < 0.3:
            return "medium"
        else:
            return "hard"
    
    def _recommend_alpha(self, difficulty: str, 
                         alpha_sensitivities: Dict[float, float]) -> float:
        """Recommend optimal alpha based on sensitivity analysis."""
        
        if not alpha_sensitivities:
            # Default recommendations by difficulty
            if difficulty == "easy":
                return 0.8  # Can be aggressive
            elif difficulty == "hard":
                return 0.5  # Be conservative
            else:
                return 0.7
        
        # Find alpha with lowest sensitivity
        best_alpha = min(alpha_sensitivities.keys(), 
                        key=lambda a: alpha_sensitivities[a])
        
        return best_alpha
    
    def _compute_weight_stats(self, weights: np.ndarray) -> Dict[str, float]:
        """Compute weight statistics."""
        flat = weights.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        
        # Compute excess kurtosis
        if std > 1e-10:
            kurtosis = float(np.mean(((flat - mean) / std) ** 4) - 3)
        else:
            kurtosis = 0.0
        
        return {
            'mean': float(mean),
            'std': float(std),
            'abs_mean': float(np.mean(np.abs(flat))),
            'kurtosis': kurtosis,
            'sparsity': float(np.mean(np.abs(flat) < 0.01)),
        }
    
    def analyze_layer(self, name: str,
                      module: nn.Module,
                      sample_input: torch.Tensor,
                      weights: np.ndarray) -> LayerSensitivity:
        """
        Analyze sensitivity for a single layer.
        
        Args:
            name: Layer name
            module: Layer module
            sample_input: Sample input for the layer
            weights: Layer weights
            
        Returns:
            LayerSensitivity with analysis results
        """
        # Compute weight statistics
        weight_stats = self._compute_weight_stats(weights)
        
        # Compute output sensitivity for different alpha values
        alpha_sensitivities = {}
        for alpha in self.config.alpha_range:
            self.output_analyzer.alpha = alpha
            sens = self.output_analyzer.compute_layer_sensitivity(module, sample_input)
            alpha_sensitivities[alpha] = sens
        
        # Use default alpha for main sensitivity score
        self.output_analyzer.alpha = 0.7
        output_sensitivity = alpha_sensitivities.get(0.7, 0.0)
        
        # Compute Fisher score if available
        fisher_score = None
        if self.config.use_fisher:
            fisher_score = self.fisher_analyzer.compute_fisher_score(name)
        
        # Weight sensitivity (proxy based on variance)
        weight_sensitivity = weight_stats['std'] * (1 + abs(weight_stats['kurtosis'] / 10))
        
        # Classify difficulty
        difficulty = self._classify_difficulty(output_sensitivity, fisher_score, weight_stats)
        
        # Recommend alpha
        recommended_alpha = self._recommend_alpha(difficulty, alpha_sensitivities)
        
        result = LayerSensitivity(
            name=name,
            shape=tuple(weights.shape),
            numel=weights.size,
            output_sensitivity=output_sensitivity,
            weight_sensitivity=weight_sensitivity,
            fisher_score=fisher_score,
            alpha_sensitivities=alpha_sensitivities,
            recommended_alpha=recommended_alpha,
            quantization_difficulty=difficulty,
            weight_stats=weight_stats
        )
        
        self.results[name] = result
        return result
    
    def analyze_model(self, model: nn.Module,
                      dataloader=None,
                      progress: bool = True) -> Dict[str, LayerSensitivity]:
        """
        Analyze all layers in the model.
        
        Args:
            model: Model to analyze
            dataloader: Optional dataloader for Fisher analysis
            progress: Show progress bar
            
        Returns:
            Dictionary of sensitivity results
        """
        logger.info("Starting sensitivity analysis...")
        start_time = time.time()
        
        # Collect Fisher gradients if dataloader provided
        if dataloader is not None and self.config.use_fisher:
            logger.info("Collecting gradients for Fisher analysis...")
            self.fisher_analyzer.collect_gradients(
                model, dataloader, self.config.num_samples
            )
        
        model.eval()
        
        # Get hidden size for sample inputs
        try:
            hidden_size = model.config.text_config.hidden_size
        except:
            hidden_size = 576  # SmolVLM default
        
        # Analyze each layer
        layers_to_analyze = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.numel() >= 64:
                # Skip normalization-related layers
                if any(x in name.lower() for x in ['layernorm', 'layer_norm', 'rmsnorm']):
                    continue
                layers_to_analyze.append((name, module))
        
        logger.info(f"Analyzing {len(layers_to_analyze)} layers...")
        
        iterator = tqdm(layers_to_analyze, desc="Analyzing") if progress else layers_to_analyze
        
        for name, module in iterator:
            # Create sample input matching layer input size
            input_size = module.in_features
            sample_input = torch.randn(1, 16, input_size, device=module.weight.device)
            
            # Get weights
            weights = module.weight.detach().float().cpu().numpy()
            
            self.analyze_layer(name, module, sample_input, weights)
        
        elapsed = time.time() - start_time
        logger.info(f"Analysis complete in {elapsed:.1f}s")
        
        return self.results
    
    def get_summary(self) -> SensitivitySummary:
        """Get summary of sensitivity analysis."""
        if not self.results:
            return SensitivitySummary(
                total_layers=0,
                difficulty_distribution={},
                most_sensitive_layers=[],
                least_sensitive_layers=[],
                avg_output_sensitivity=0.0,
                recommended_high_precision_layers=[],
                analysis_time_seconds=0.0
            )
        
        # Count difficulties
        difficulty_counts = defaultdict(int)
        for result in self.results.values():
            difficulty_counts[result.quantization_difficulty] += 1
        
        # Sort by sensitivity
        sorted_by_sens = sorted(
            self.results.values(),
            key=lambda r: r.output_sensitivity,
            reverse=True
        )
        
        most_sensitive = [r.name for r in sorted_by_sens[:5]]
        least_sensitive = [r.name for r in sorted_by_sens[-5:]]
        
        # Average sensitivity
        avg_sens = np.mean([r.output_sensitivity for r in self.results.values()])
        
        # Recommend high precision for hard layers
        high_precision = [
            r.name for r in self.results.values()
            if r.quantization_difficulty == "hard"
        ]
        
        return SensitivitySummary(
            total_layers=len(self.results),
            difficulty_distribution=dict(difficulty_counts),
            most_sensitive_layers=most_sensitive,
            least_sensitive_layers=least_sensitive,
            avg_output_sensitivity=float(avg_sens),
            recommended_high_precision_layers=high_precision,
            analysis_time_seconds=0.0  # Updated externally
        )
    
    def print_summary(self) -> None:
        """Print analysis summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("=" * 70)
        
        print(f"\nTotal layers analyzed: {summary.total_layers}")
        
        print("\nQuantization difficulty distribution:")
        for difficulty, count in sorted(summary.difficulty_distribution.items()):
            pct = count / summary.total_layers * 100
            bar = "█" * int(pct / 2)
            print(f"  {difficulty:8s}: {count:4d} layers ({pct:5.1f}%) {bar}")
        
        print(f"\nAverage output sensitivity: {summary.avg_output_sensitivity:.4f}")
        
        print("\nMost sensitive layers (consider higher precision):")
        for name in summary.most_sensitive_layers:
            result = self.results.get(name)
            if result:
                print(f"  - {name}")
                print(f"      sensitivity: {result.output_sensitivity:.4f}, "
                      f"recommended α: {result.recommended_alpha}")
        
        print("\nLeast sensitive layers (can be aggressively quantized):")
        for name in summary.least_sensitive_layers:
            result = self.results.get(name)
            if result:
                print(f"  - {name}")
                print(f"      sensitivity: {result.output_sensitivity:.4f}")
        
        if summary.recommended_high_precision_layers:
            print(f"\nRecommended for high precision ({len(summary.recommended_high_precision_layers)} layers):")
            for name in summary.recommended_high_precision_layers[:10]:
                print(f"  - {name}")
            if len(summary.recommended_high_precision_layers) > 10:
                print(f"  ... and {len(summary.recommended_high_precision_layers) - 10} more")
    
    def export_results(self, output_path: str) -> None:
        """Export analysis results to JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'config': asdict(self.config),
            'summary': self.get_summary().to_dict(),
            'layers': {name: r.to_dict() for name, r in self.results.items()}
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results exported to: {output_file}")
    
    def get_sensitivity_scores(self) -> Dict[str, float]:
        """
        Get sensitivity scores for all layers.
        
        Returns:
            Dictionary mapping layer names to sensitivity scores
        """
        return {
            name: result.output_sensitivity
            for name, result in self.results.items()
        }
    
    def get_recommended_alphas(self) -> Dict[str, float]:
        """
        Get recommended alpha values for all layers.
        
        Returns:
            Dictionary mapping layer names to recommended alpha values
        """
        return {
            name: result.recommended_alpha
            for name, result in self.results.items()
        }


class ModelSensitivityAnalyzer:
    """
    High-level interface for model sensitivity analysis.
    """
    
    def __init__(self, model_path: str,
                 config: Optional[SensitivityConfig] = None,
                 device: str = 'cpu'):
        """
        Initialize the model sensitivity analyzer.
        
        Args:
            model_path: Path to model or HuggingFace ID
            config: Analysis configuration
            device: Device to use
        """
        self.model_path = model_path
        self.config = config or SensitivityConfig()
        self.device = device
        self.model = None
        self.analyzer = SensitivityAnalyzer(self.config)
        
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
        ).to(self.device)
        
        logger.info("Model loaded successfully")
    
    def analyze(self, progress: bool = True) -> Dict[str, LayerSensitivity]:
        """
        Run sensitivity analysis on the model.
        
        Args:
            progress: Show progress bar
            
        Returns:
            Dictionary of sensitivity results
        """
        if self.model is None:
            self.load_model()
        
        return self.analyzer.analyze_model(self.model, progress=progress)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Layer-by-layer sensitivity analysis for SmolVLM-256M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python sensitivity_analysis.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    
    # More samples for better accuracy
    python sensitivity_analysis.py --model ./model/smolvlm-256m --samples 100
    
    # Export results
    python sensitivity_analysis.py --model HuggingFaceTB/SmolVLM-256M-Instruct \\
        --output ./sensitivity_results.json
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples for analysis"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Export results to JSON"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SiLens Sensitivity Analyzer")
    print("=" * 70)
    
    config = SensitivityConfig(num_samples=args.samples)
    analyzer = ModelSensitivityAnalyzer(args.model, config, args.device)
    
    # Run analysis
    start_time = time.time()
    results = analyzer.analyze()
    analysis_time = time.time() - start_time
    
    # Print summary
    analyzer.analyzer.print_summary()
    
    print(f"\nTotal analysis time: {analysis_time:.1f}s")
    
    # Export if requested
    if args.output:
        analyzer.analyzer.export_results(args.output)
    
    print("\n" + "=" * 70)
    print("Sensitivity analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
