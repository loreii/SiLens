#!/usr/bin/env python3
"""
Sparse Attention Pattern Analysis for SiLens.

This module analyzes attention patterns to identify opportunities for
sparse attention optimizations in hardware. Sparse attention can
significantly reduce memory bandwidth and compute requirements.

Supported Patterns:
- Local/Sliding Window: Attend only to nearby tokens
- Strided: Attend to every k-th token
- Block Sparse: Fixed block patterns
- Dynamic Sparse: Learned/adaptive sparsity
- Cross-Modal: Different patterns for vision vs text

Usage:
    python sparse_attention.py \\
        --model HuggingFaceTB/SmolVLM-256M-Instruct \\
        --dataset ./calibration_data \\
        --output ./attention_analysis

Author: SiLens AI/ML Team
License: Apache 2.0
"""

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SparsePattern(Enum):
    """Types of sparse attention patterns."""
    DENSE = "dense"              # Full attention (baseline)
    LOCAL = "local"              # Sliding window
    STRIDED = "strided"          # Fixed stride pattern
    BLOCK_SPARSE = "block"       # Block-wise sparsity
    LONGFORMER = "longformer"    # Local + global tokens
    BIGBIRD = "bigbird"          # Local + global + random
    LINEAR = "linear"            # Linear attention approximation


@dataclass
class AttentionStats:
    """Statistics for attention patterns in a layer."""
    layer_name: str
    layer_type: str  # 'vision', 'language', 'cross'
    
    # Basic stats
    num_heads: int
    seq_len: int
    head_dim: int
    
    # Sparsity analysis
    mean_entropy: float              # Average attention entropy
    mean_sparsity: float             # Fraction of near-zero weights
    effective_context_len: float     # Effective number of attended tokens
    
    # Pattern analysis
    local_concentration: float       # Attention mass in local window
    diagonal_strength: float         # Strength of diagonal pattern
    global_token_mass: float         # Mass on special tokens (CLS, etc.)
    
    # Recommended pattern
    recommended_pattern: SparsePattern = SparsePattern.DENSE
    recommended_params: Dict[str, Any] = field(default_factory=dict)
    
    # Memory/compute savings
    dense_ops: int = 0
    sparse_ops: int = 0
    reduction_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['recommended_pattern'] = self.recommended_pattern.value
        return d


@dataclass
class SparsePatternConfig:
    """Configuration for a sparse attention pattern."""
    pattern: SparsePattern
    window_size: int = 256         # For local attention
    stride: int = 64               # For strided attention
    block_size: int = 64           # For block sparse
    num_global_tokens: int = 1     # Global tokens (CLS, etc.)
    num_random_tokens: int = 0     # Random tokens (BigBird)
    
    def compute_sparsity(self, seq_len: int) -> float:
        """Compute sparsity ratio for given sequence length."""
        if self.pattern == SparsePattern.DENSE:
            return 0.0
        
        elif self.pattern == SparsePattern.LOCAL:
            # Local window: each token attends to window_size tokens
            attended = min(self.window_size, seq_len)
            return 1.0 - (attended / seq_len)
        
        elif self.pattern == SparsePattern.STRIDED:
            # Strided: attend to every stride-th token
            attended = seq_len // self.stride + self.window_size
            return 1.0 - (min(attended, seq_len) / seq_len)
        
        elif self.pattern == SparsePattern.BLOCK_SPARSE:
            # Block sparse: attend to adjacent blocks
            num_blocks = seq_len // self.block_size
            attended_blocks = 3  # Adjacent + same block
            return 1.0 - (attended_blocks / num_blocks) if num_blocks > 0 else 0.0
        
        elif self.pattern == SparsePattern.LONGFORMER:
            # Local + global tokens
            local = min(self.window_size, seq_len)
            global_tokens = self.num_global_tokens
            attended = local + global_tokens
            return 1.0 - (min(attended, seq_len) / seq_len)
        
        return 0.0


class AttentionAnalyzer:
    """
    Analyze attention patterns to recommend sparse patterns.
    
    Collects attention weights during inference and analyzes
    them to determine optimal sparse patterns for each layer.
    """
    
    def __init__(self, sparsity_threshold: float = 0.01,
                 local_window: int = 128,
                 min_samples: int = 10):
        """
        Initialize analyzer.
        
        Args:
            sparsity_threshold: Threshold below which attention is "sparse"
            local_window: Window size for local attention analysis
            min_samples: Minimum samples for statistical significance
        """
        self.sparsity_threshold = sparsity_threshold
        self.local_window = local_window
        self.min_samples = min_samples
        
        self.attention_samples: Dict[str, List[np.ndarray]] = {}
        self.layer_stats: Dict[str, AttentionStats] = {}
    
    def collect_attention(self, layer_name: str, 
                          attention_weights: torch.Tensor) -> None:
        """
        Collect attention weights for analysis.
        
        Args:
            layer_name: Name of the attention layer
            attention_weights: Attention weights [batch, heads, seq, seq]
        """
        weights = attention_weights.detach().cpu().numpy()
        
        if layer_name not in self.attention_samples:
            self.attention_samples[layer_name] = []
        
        self.attention_samples[layer_name].append(weights)
    
    def analyze_layer(self, layer_name: str,
                      layer_type: str = 'language') -> AttentionStats:
        """
        Analyze collected attention samples for a layer.
        
        Args:
            layer_name: Name of the layer
            layer_type: Type of layer ('vision', 'language', 'cross')
            
        Returns:
            AttentionStats with analysis results
        """
        if layer_name not in self.attention_samples:
            raise ValueError(f"No samples for layer: {layer_name}")
        
        samples = self.attention_samples[layer_name]
        if len(samples) < self.min_samples:
            logger.warning(f"Only {len(samples)} samples for {layer_name}")
        
        # Stack all samples
        all_weights = np.concatenate(samples, axis=0)
        # Shape: [total_batch, heads, seq, seq]
        
        num_heads = all_weights.shape[1]
        seq_len = all_weights.shape[2]
        
        # Compute statistics
        mean_entropy = self._compute_entropy(all_weights)
        mean_sparsity = self._compute_sparsity(all_weights)
        effective_context = self._compute_effective_context(all_weights)
        local_concentration = self._compute_local_concentration(all_weights)
        diagonal_strength = self._compute_diagonal_strength(all_weights)
        global_mass = self._compute_global_token_mass(all_weights)
        
        # Determine recommended pattern
        pattern, params = self._recommend_pattern(
            mean_sparsity, local_concentration, diagonal_strength,
            effective_context, seq_len, layer_type
        )
        
        # Compute ops savings
        dense_ops = seq_len * seq_len * num_heads
        sparse_config = SparsePatternConfig(pattern=pattern, **params)
        sparsity = sparse_config.compute_sparsity(seq_len)
        sparse_ops = int(dense_ops * (1 - sparsity))
        
        stats = AttentionStats(
            layer_name=layer_name,
            layer_type=layer_type,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=64,  # Assumed
            mean_entropy=float(mean_entropy),
            mean_sparsity=float(mean_sparsity),
            effective_context_len=float(effective_context),
            local_concentration=float(local_concentration),
            diagonal_strength=float(diagonal_strength),
            global_token_mass=float(global_mass),
            recommended_pattern=pattern,
            recommended_params=params,
            dense_ops=dense_ops,
            sparse_ops=sparse_ops,
            reduction_ratio=sparse_ops / dense_ops if dense_ops > 0 else 1.0
        )
        
        self.layer_stats[layer_name] = stats
        return stats
    
    def _compute_entropy(self, weights: np.ndarray) -> float:
        """Compute average entropy of attention distributions."""
        # Clip for numerical stability
        weights = np.clip(weights, 1e-10, 1.0)
        
        # Entropy: -sum(p * log(p))
        entropy = -np.sum(weights * np.log(weights), axis=-1)
        return np.mean(entropy)
    
    def _compute_sparsity(self, weights: np.ndarray) -> float:
        """Compute fraction of near-zero attention weights."""
        sparse_mask = weights < self.sparsity_threshold
        return np.mean(sparse_mask)
    
    def _compute_effective_context(self, weights: np.ndarray) -> float:
        """
        Compute effective context length.
        
        Uses the inverse of the sum of squared probabilities:
        effective_n = 1 / sum(p_i^2)
        """
        sum_squared = np.sum(weights ** 2, axis=-1)
        effective_n = 1.0 / (sum_squared + 1e-10)
        return np.mean(effective_n)

    def _compute_local_concentration(self, weights: np.ndarray) -> float:
        """Compute attention mass within local window around diagonal."""
        seq_len = weights.shape[-1]
        
        # Create local window mask
        indices = np.arange(seq_len)
        row_idx = indices[:, None]
        col_idx = indices[None, :]
        
        local_mask = np.abs(row_idx - col_idx) <= (self.local_window // 2)
        
        # Compute mass in local window
        # weights: [batch, heads, seq, seq]
        local_mass = np.sum(weights * local_mask, axis=-1)
        return np.mean(local_mass)
    
    def _compute_diagonal_strength(self, weights: np.ndarray) -> float:
        """Compute strength of diagonal attention (self-attention on same position)."""
        # Extract diagonal elements
        seq_len = weights.shape[-1]
        diag_indices = np.arange(seq_len)
        
        # weights[..., i, i] for all i
        diagonal_weights = weights[..., diag_indices, diag_indices]
        return np.mean(diagonal_weights)
    
    def _compute_global_token_mass(self, weights: np.ndarray) -> float:
        """Compute attention mass on first few tokens (often special tokens)."""
        # First token (usually CLS or BOS)
        first_token_mass = weights[..., :, 0]
        return np.mean(first_token_mass)
    
    def _recommend_pattern(self, sparsity: float, local_conc: float,
                           diag_strength: float, effective_ctx: float,
                           seq_len: int, layer_type: str) -> Tuple[SparsePattern, Dict]:
        """
        Recommend sparse pattern based on attention statistics.
        
        Returns:
            Tuple of (pattern, parameters)
        """
        params = {}
        
        # Vision attention: often needs full attention for spatial relationships
        if layer_type == 'vision':
            if local_conc > 0.7:
                # Strong local patterns - use local attention
                window = max(64, int(effective_ctx * 2))
                return SparsePattern.LOCAL, {'window_size': window}
            else:
                return SparsePattern.DENSE, {}
        
        # Language/cross attention
        
        # Very sparse with local patterns
        if sparsity > 0.8 and local_conc > 0.6:
            window = max(64, int(effective_ctx * 1.5))
            return SparsePattern.LOCAL, {'window_size': window}
        
        # Moderate sparsity with some global tokens
        if sparsity > 0.6 and local_conc > 0.4:
            window = max(128, int(effective_ctx * 2))
            return SparsePattern.LONGFORMER, {
                'window_size': window,
                'num_global_tokens': max(1, int(seq_len * 0.01))
            }
        
        # Low sparsity but long sequences
        if seq_len > 2048:
            # Use block sparse for very long sequences
            block_size = 64
            return SparsePattern.BLOCK_SPARSE, {'block_size': block_size}
        
        # Default to dense for short sequences or complex patterns
        return SparsePattern.DENSE, {}
    
    def analyze_all_layers(self) -> Dict[str, AttentionStats]:
        """Analyze all collected layers."""
        for layer_name in self.attention_samples:
            # Infer layer type from name
            if 'vision' in layer_name.lower() or 'vit' in layer_name.lower():
                layer_type = 'vision'
            elif 'cross' in layer_name.lower():
                layer_type = 'cross'
            else:
                layer_type = 'language'
            
            self.analyze_layer(layer_name, layer_type)
        
        return self.layer_stats
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all layer analyses."""
        if not self.layer_stats:
            return {}
        
        pattern_counts = {}
        total_dense_ops = 0
        total_sparse_ops = 0
        
        for stats in self.layer_stats.values():
            pattern = stats.recommended_pattern.value
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            total_dense_ops += stats.dense_ops
            total_sparse_ops += stats.sparse_ops
        
        overall_reduction = total_sparse_ops / total_dense_ops if total_dense_ops > 0 else 1.0
        
        return {
            'num_layers': len(self.layer_stats),
            'pattern_distribution': pattern_counts,
            'total_dense_ops': total_dense_ops,
            'total_sparse_ops': total_sparse_ops,
            'overall_compute_reduction': overall_reduction,
            'memory_reduction': overall_reduction,  # Approximate
            'avg_sparsity': np.mean([s.mean_sparsity for s in self.layer_stats.values()]),
            'avg_effective_context': np.mean([s.effective_context_len for s in self.layer_stats.values()])
        }
    
    def export_config(self, output_path: str) -> None:
        """Export sparse attention configuration for hardware."""
        config = {
            'summary': self.get_summary(),
            'layers': {name: stats.to_dict() for name, stats in self.layer_stats.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Exported configuration to {output_path}")
    
    def print_report(self) -> None:
        """Print analysis report."""
        print("\n" + "=" * 70)
        print("SPARSE ATTENTION ANALYSIS REPORT")
        print("=" * 70)
        
        summary = self.get_summary()
        
        print(f"\nLayers Analyzed: {summary.get('num_layers', 0)}")
        print(f"\nPattern Distribution:")
        for pattern, count in summary.get('pattern_distribution', {}).items():
            print(f"  {pattern}: {count} layers")
        
        print(f"\nCompute Reduction: {summary.get('overall_compute_reduction', 1.0):.2%}")
        print(f"Memory Reduction:  {summary.get('memory_reduction', 1.0):.2%}")
        print(f"Average Sparsity:  {summary.get('avg_sparsity', 0.0):.2%}")
        print(f"Avg Effective Context: {summary.get('avg_effective_context', 0.0):.1f} tokens")
        
        print(f"\n{'Layer':<40} {'Pattern':<12} {'Reduction':<10}")
        print("-" * 62)
        
        for name, stats in sorted(self.layer_stats.items()):
            print(f"{name[:40]:<40} {stats.recommended_pattern.value:<12} "
                  f"{stats.reduction_ratio:.2%}")
        
        print("=" * 70)


def generate_sparse_mask(pattern: SparsePatternConfig, 
                          seq_len: int) -> np.ndarray:
    """
    Generate sparse attention mask for given pattern.
    
    Args:
        pattern: Sparse pattern configuration
        seq_len: Sequence length
        
    Returns:
        Boolean mask [seq_len, seq_len] where True = attend
    """
    mask = np.zeros((seq_len, seq_len), dtype=bool)
    
    if pattern.pattern == SparsePattern.DENSE:
        mask[:] = True
    
    elif pattern.pattern == SparsePattern.LOCAL:
        half_window = pattern.window_size // 2
        for i in range(seq_len):
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            mask[i, start:end] = True
    
    elif pattern.pattern == SparsePattern.STRIDED:
        half_window = pattern.window_size // 2
        for i in range(seq_len):
            # Local window
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            mask[i, start:end] = True
            
            # Strided global
            for j in range(0, seq_len, pattern.stride):
                mask[i, j] = True
    
    elif pattern.pattern == SparsePattern.LONGFORMER:
        half_window = pattern.window_size // 2
        for i in range(seq_len):
            # Local window
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            mask[i, start:end] = True
        
        # Global tokens attend to/from all
        for g in range(pattern.num_global_tokens):
            if g < seq_len:
                mask[g, :] = True
                mask[:, g] = True
    
    elif pattern.pattern == SparsePattern.BLOCK_SPARSE:
        block_size = pattern.block_size
        for i in range(seq_len):
            block_i = i // block_size
            # Same block
            start = block_i * block_size
            end = min((block_i + 1) * block_size, seq_len)
            mask[i, start:end] = True
            
            # Previous block
            if block_i > 0:
                prev_start = (block_i - 1) * block_size
                prev_end = block_i * block_size
                mask[i, prev_start:prev_end] = True
    
    return mask


def main():
    parser = argparse.ArgumentParser(description='Sparse Attention Analysis')
    parser.add_argument('--output', default='./attention_analysis.json')
    parser.add_argument('--seq-len', type=int, default=2048)
    
    args = parser.parse_args()
    
    # Demo: generate sample analysis
    analyzer = AttentionAnalyzer()
    
    # Generate synthetic attention patterns
    for layer_idx in range(12):
        layer_name = f"model.layers.{layer_idx}.self_attn"
        
        # Simulate attention with varying sparsity
        weights = np.random.dirichlet(np.ones(args.seq_len) * (12 - layer_idx + 1), 
                                       size=(4, 8, args.seq_len))
        
        analyzer.collect_attention(layer_name, torch.from_numpy(weights))
    
    analyzer.analyze_all_layers()
    analyzer.print_report()
    analyzer.export_config(args.output)


if __name__ == '__main__':
    main()
