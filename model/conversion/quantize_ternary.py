#!/usr/bin/env python3
"""
Ternary quantization for SmolVLM-256M weights.

This module implements ternary quantization (-1, 0, +1) for neural network weights,
optimized for hardware implementation on the SiLens accelerator.

Quantization Formula:
    q(w) = +1  if w > threshold
         = -1  if w < -threshold
         =  0  otherwise
    
    where: threshold = α * mean(|w|), typically α ∈ [0.5, 0.9]

Hardware Encoding:
    +1 → 0b01 (connect to VDD)
    -1 → 0b10 (connect to GND)  
     0 → 0b00 (no connection / high-Z)

Features:
- Per-layer and per-channel quantization
- Configurable threshold factor (α)
- Scale factor computation for activation scaling
- Comprehensive statistics on quantization quality
- Export to hardware-friendly formats

Usage:
    python quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    python quantize_ternary.py --model ./model/smolvlm-256m --alpha 0.7 --output ./quantized

Author: SiLens AI/ML Team
License: Apache 2.0
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantizationMode(Enum):
    """Quantization granularity modes."""
    PER_TENSOR = "per_tensor"      # Single threshold for entire tensor
    PER_CHANNEL = "per_channel"    # Separate threshold per output channel
    PER_GROUP = "per_group"        # Group-wise quantization (e.g., 128 weights)


@dataclass
class TernaryQuantizationConfig:
    """Configuration for ternary quantization."""
    alpha: float = 0.7                           # Threshold factor
    mode: QuantizationMode = QuantizationMode.PER_TENSOR
    group_size: int = 128                        # For per-group mode
    symmetric: bool = True                       # Symmetric quantization
    skip_normalization: bool = True              # Skip LayerNorm/RMSNorm weights
    skip_embeddings: bool = False                # Skip embedding weights


@dataclass 
class QuantizationResult:
    """Result of quantizing a single tensor."""
    name: str
    original_shape: Tuple[int, ...]
    quantized_weights: np.ndarray       # int8: -1, 0, +1
    scale_factors: np.ndarray           # For dequantization
    thresholds: np.ndarray              # Thresholds used
    
    # Statistics
    num_positive: int
    num_negative: int
    num_zero: int
    sparsity: float                     # Fraction of zeros
    mean_abs_error: float               # Quantization error
    max_abs_error: float
    
    # Hardware encoding
    encoded_weights: Optional[np.ndarray] = None  # 2-bit packed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'name': self.name,
            'original_shape': list(self.original_shape),
            'num_positive': self.num_positive,
            'num_negative': self.num_negative,
            'num_zero': self.num_zero,
            'sparsity': self.sparsity,
            'mean_abs_error': self.mean_abs_error,
            'max_abs_error': self.max_abs_error,
        }


class TernaryQuantizer:
    """
    Ternary weight quantizer for neural networks.
    
    Implements the quantization scheme:
        q(w) = sign(w) * 1_{|w| > threshold}
    
    where threshold = α * mean(|w|) for each quantization group.
    """
    
    def __init__(self, config: Optional[TernaryQuantizationConfig] = None):
        """
        Initialize the quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config or TernaryQuantizationConfig()
        self.results: Dict[str, QuantizationResult] = {}
        
    def compute_threshold(self, weights: np.ndarray, 
                          mode: QuantizationMode,
                          alpha: float) -> np.ndarray:
        """
        Compute quantization threshold(s).
        
        Args:
            weights: Weight tensor
            mode: Quantization mode
            alpha: Threshold factor
            
        Returns:
            Threshold value(s) as ndarray
        """
        if mode == QuantizationMode.PER_TENSOR:
            # Single threshold for entire tensor
            threshold = alpha * np.mean(np.abs(weights))
            return np.array([threshold])
        
        elif mode == QuantizationMode.PER_CHANNEL:
            # One threshold per output channel (first dimension)
            abs_weights = np.abs(weights)
            if weights.ndim >= 2:
                # Reshape to (out_features, -1) and compute mean
                reshaped = abs_weights.reshape(weights.shape[0], -1)
                thresholds = alpha * np.mean(reshaped, axis=1)
            else:
                thresholds = np.array([alpha * np.mean(abs_weights)])
            return thresholds
        
        elif mode == QuantizationMode.PER_GROUP:
            # Group-wise thresholds
            flat = weights.flatten()
            num_groups = (len(flat) + self.config.group_size - 1) // self.config.group_size
            
            thresholds = []
            for i in range(num_groups):
                start = i * self.config.group_size
                end = min(start + self.config.group_size, len(flat))
                group = flat[start:end]
                thresholds.append(alpha * np.mean(np.abs(group)))
            
            return np.array(thresholds)
        
        else:
            raise ValueError(f"Unknown quantization mode: {mode}")
    
    def quantize_tensor(self, name: str, 
                        weights: np.ndarray,
                        mode: Optional[QuantizationMode] = None,
                        alpha: Optional[float] = None) -> QuantizationResult:
        """
        Quantize a single weight tensor to ternary values.
        
        Args:
            name: Name of the weight tensor
            weights: Weight values as numpy array
            mode: Override quantization mode
            alpha: Override threshold factor
            
        Returns:
            QuantizationResult with quantized weights and statistics
        """
        mode = mode or self.config.mode
        alpha = alpha if alpha is not None else self.config.alpha
        
        original_shape = weights.shape
        flat_weights = weights.flatten().astype(np.float32)
        
        # Compute thresholds
        thresholds = self.compute_threshold(weights, mode, alpha)
        
        # Apply quantization
        if mode == QuantizationMode.PER_TENSOR:
            threshold = thresholds[0]
            quantized = np.zeros_like(flat_weights, dtype=np.int8)
            quantized[flat_weights > threshold] = 1
            quantized[flat_weights < -threshold] = -1
            scale = np.mean(np.abs(flat_weights[np.abs(flat_weights) > threshold])) if np.any(np.abs(flat_weights) > threshold) else 1.0
            scale_factors = np.array([scale])
            
        elif mode == QuantizationMode.PER_CHANNEL:
            quantized = np.zeros_like(flat_weights, dtype=np.int8)
            scale_factors = np.zeros(len(thresholds))
            
            if weights.ndim >= 2:
                reshaped = weights.reshape(weights.shape[0], -1)
                quantized_reshaped = np.zeros_like(reshaped, dtype=np.int8)
                
                for i, threshold in enumerate(thresholds):
                    row = reshaped[i]
                    quantized_reshaped[i][row > threshold] = 1
                    quantized_reshaped[i][row < -threshold] = -1
                    
                    # Compute scale for this channel
                    active_mask = np.abs(row) > threshold
                    if np.any(active_mask):
                        scale_factors[i] = np.mean(np.abs(row[active_mask]))
                    else:
                        scale_factors[i] = 1.0
                
                quantized = quantized_reshaped.flatten()
            else:
                threshold = thresholds[0]
                quantized[flat_weights > threshold] = 1
                quantized[flat_weights < -threshold] = -1
                scale_factors = np.array([np.mean(np.abs(flat_weights)) if np.mean(np.abs(flat_weights)) > 0 else 1.0])
        
        elif mode == QuantizationMode.PER_GROUP:
            quantized = np.zeros_like(flat_weights, dtype=np.int8)
            scale_factors = np.zeros(len(thresholds))
            
            for i, threshold in enumerate(thresholds):
                start = i * self.config.group_size
                end = min(start + self.config.group_size, len(flat_weights))
                group = flat_weights[start:end]
                
                quantized[start:end][group > threshold] = 1
                quantized[start:end][group < -threshold] = -1
                
                active_mask = np.abs(group) > threshold
                if np.any(active_mask):
                    scale_factors[i] = np.mean(np.abs(group[active_mask]))
                else:
                    scale_factors[i] = 1.0
        
        # Reshape back to original shape
        quantized = quantized.reshape(original_shape)
        
        # Compute statistics
        num_positive = int(np.sum(quantized == 1))
        num_negative = int(np.sum(quantized == -1))
        num_zero = int(np.sum(quantized == 0))
        total = quantized.size
        
        # Compute quantization error
        if mode == QuantizationMode.PER_TENSOR:
            dequantized = quantized.astype(np.float32) * scale_factors[0]
        elif mode == QuantizationMode.PER_CHANNEL:
            if weights.ndim >= 2:
                dequantized = quantized.astype(np.float32)
                for i, scale in enumerate(scale_factors):
                    dequantized[i] *= scale
            else:
                dequantized = quantized.astype(np.float32) * scale_factors[0]
        else:  # PER_GROUP
            dequantized = quantized.flatten().astype(np.float32)
            for i, scale in enumerate(scale_factors):
                start = i * self.config.group_size
                end = min(start + self.config.group_size, len(dequantized))
                dequantized[start:end] *= scale
            dequantized = dequantized.reshape(original_shape)
        
        error = np.abs(weights - dequantized)
        
        result = QuantizationResult(
            name=name,
            original_shape=original_shape,
            quantized_weights=quantized,
            scale_factors=scale_factors,
            thresholds=thresholds,
            num_positive=num_positive,
            num_negative=num_negative,
            num_zero=num_zero,
            sparsity=num_zero / total,
            mean_abs_error=float(np.mean(error)),
            max_abs_error=float(np.max(error)),
        )
        
        self.results[name] = result
        return result
    
    def encode_for_hardware(self, quantized: np.ndarray) -> np.ndarray:
        """
        Encode ternary weights for hardware implementation.
        
        Encoding scheme (2 bits per weight):
            +1 → 0b01
            -1 → 0b10
             0 → 0b00
        
        Packs 4 weights per byte.
        
        Args:
            quantized: Ternary weights (-1, 0, +1)
            
        Returns:
            Packed uint8 array
        """
        flat = quantized.flatten()
        
        # Pad to multiple of 4
        pad_len = (4 - len(flat) % 4) % 4
        if pad_len > 0:
            flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.int8)])
        
        # Convert to 2-bit encoding
        encoded = np.zeros(len(flat), dtype=np.uint8)
        encoded[flat == 1] = 0b01
        encoded[flat == -1] = 0b10
        # 0 stays as 0b00
        
        # Pack 4 weights per byte
        packed = (encoded[0::4] << 6) | (encoded[1::4] << 4) | (encoded[2::4] << 2) | encoded[3::4]
        
        return packed
    
    def should_skip_layer(self, name: str) -> bool:
        """Determine if a layer should be skipped from quantization."""
        name_lower = name.lower()
        
        if self.config.skip_normalization:
            if any(x in name_lower for x in ['layernorm', 'layer_norm', 'rmsnorm', 'norm.weight']):
                return True
        
        if self.config.skip_embeddings:
            if any(x in name_lower for x in ['embed', 'wte', 'wpe']):
                return True
        
        # Always skip biases (usually small tensors that don't benefit from quantization)
        if name_lower.endswith('.bias'):
            return True
        
        return False


class ModelQuantizer:
    """
    High-level quantizer for SmolVLM-256M model.
    
    Handles:
    - Loading model
    - Quantizing all applicable weights
    - Computing aggregate statistics
    - Exporting quantized model
    """
    
    def __init__(self, model_path: str, 
                 config: Optional[TernaryQuantizationConfig] = None,
                 device: str = 'cpu'):
        """
        Initialize the model quantizer.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            config: Quantization configuration
            device: Device to load model on
        """
        self.model_path = model_path
        self.config = config or TernaryQuantizationConfig()
        self.device = device
        self.model = None
        self.quantizer = TernaryQuantizer(self.config)
        self.results: Dict[str, QuantizationResult] = {}
        
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
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {total_params:,} parameters")
    
    def quantize_all_weights(self) -> Dict[str, QuantizationResult]:
        """
        Quantize all applicable weights in the model.
        
        Returns:
            Dictionary of quantization results
        """
        if self.model is None:
            self.load_model()
        
        logger.info("Quantizing weights...")
        
        skipped = []
        quantized_count = 0
        
        for name, param in self.model.named_parameters():
            if self.quantizer.should_skip_layer(name):
                skipped.append(name)
                continue
            
            # Skip very small tensors
            if param.numel() < 64:
                skipped.append(name)
                continue
            
            weights = param.detach().float().cpu().numpy()
            result = self.quantizer.quantize_tensor(name, weights)
            self.results[name] = result
            quantized_count += 1
        
        logger.info(f"Quantized {quantized_count} tensors, skipped {len(skipped)}")
        
        return self.results
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics across all quantized layers."""
        if not self.results:
            return {}
        
        total_params = 0
        total_positive = 0
        total_negative = 0
        total_zero = 0
        all_errors = []
        
        for result in self.results.values():
            numel = np.prod(result.original_shape)
            total_params += numel
            total_positive += result.num_positive
            total_negative += result.num_negative
            total_zero += result.num_zero
            all_errors.append(result.mean_abs_error)
        
        return {
            'total_quantized_params': total_params,
            'num_layers': len(self.results),
            'distribution': {
                'positive': total_positive,
                'negative': total_negative,
                'zero': total_zero,
                'positive_pct': total_positive / total_params * 100,
                'negative_pct': total_negative / total_params * 100,
                'zero_pct': total_zero / total_params * 100,
            },
            'overall_sparsity': total_zero / total_params,
            'avg_quantization_error': float(np.mean(all_errors)),
            'max_quantization_error': float(np.max(all_errors)),
        }
    
    def print_summary(self) -> None:
        """Print quantization summary."""
        stats = self.get_summary_statistics()
        
        print("\n" + "=" * 70)
        print("TERNARY QUANTIZATION SUMMARY")
        print("=" * 70)
        
        print(f"\nConfiguration:")
        print(f"  Alpha (threshold factor): {self.config.alpha}")
        print(f"  Mode: {self.config.mode.value}")
        
        print(f"\nQuantized Layers: {stats['num_layers']}")
        print(f"Total Parameters: {stats['total_quantized_params']:,}")
        
        dist = stats['distribution']
        print(f"\nWeight Distribution:")
        print(f"  +1: {dist['positive']:,} ({dist['positive_pct']:.1f}%)")
        print(f"  -1: {dist['negative']:,} ({dist['negative_pct']:.1f}%)")
        print(f"   0: {dist['zero']:,} ({dist['zero_pct']:.1f}%)")
        
        print(f"\nSparsity: {stats['overall_sparsity']:.1%}")
        print(f"Average Quantization Error: {stats['avg_quantization_error']:.6f}")
        print(f"Max Quantization Error: {stats['max_quantization_error']:.6f}")
        
        # Memory savings
        original_bits = stats['total_quantized_params'] * 32  # FP32
        quantized_bits = stats['total_quantized_params'] * 2   # 2-bit ternary
        compression = original_bits / quantized_bits
        
        print(f"\nMemory Savings:")
        print(f"  Original (FP32): {original_bits / 8 / 1e6:.1f} MB")
        print(f"  Quantized (2-bit): {quantized_bits / 8 / 1e6:.1f} MB")
        print(f"  Compression ratio: {compression:.1f}x")
    
    def export(self, output_dir: str, 
               include_scales: bool = True,
               hardware_encoding: bool = True) -> None:
        """
        Export quantized weights.
        
        Args:
            output_dir: Output directory
            include_scales: Include scale factors
            hardware_encoding: Export hardware-packed format
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting to: {output_path}")
        
        # Export metadata
        metadata = {
            'config': {
                'alpha': self.config.alpha,
                'mode': self.config.mode.value,
                'group_size': self.config.group_size,
            },
            'summary': self.get_summary_statistics(),
            'layers': {name: r.to_dict() for name, r in self.results.items()}
        }
        
        with open(output_path / 'quantization_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Export weights
        weights_dir = output_path / 'weights'
        weights_dir.mkdir(exist_ok=True)
        
        for name, result in self.results.items():
            safe_name = name.replace('.', '_').replace('/', '_')
            
            # Save quantized weights
            np.save(weights_dir / f"{safe_name}_quantized.npy", result.quantized_weights)
            
            if include_scales:
                np.save(weights_dir / f"{safe_name}_scales.npy", result.scale_factors)
            
            if hardware_encoding:
                packed = self.quantizer.encode_for_hardware(result.quantized_weights)
                np.save(weights_dir / f"{safe_name}_packed.npy", packed)
        
        logger.info(f"Exported {len(self.results)} quantized tensors")
    
    def create_quantized_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Create a state dict with quantized weights.
        
        Useful for loading into a modified model architecture.
        
        Returns:
            State dict with quantized weights as torch tensors
        """
        state_dict = {}
        
        for name, result in self.results.items():
            # Store quantized weights as int8
            state_dict[f"{name}.quantized"] = torch.from_numpy(result.quantized_weights)
            state_dict[f"{name}.scales"] = torch.from_numpy(result.scale_factors)
        
        return state_dict


def search_optimal_alpha(model_path: str, 
                         alpha_range: List[float] = None,
                         eval_samples: int = 100) -> Dict[str, float]:
    """
    Search for optimal alpha value that minimizes quantization error.
    
    Args:
        model_path: Path to model
        alpha_range: List of alpha values to try
        eval_samples: Number of layers to evaluate
        
    Returns:
        Dictionary with optimal alpha and statistics
    """
    if alpha_range is None:
        alpha_range = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    logger.info(f"Searching optimal alpha in {alpha_range}")
    
    results = {}
    
    for alpha in alpha_range:
        config = TernaryQuantizationConfig(alpha=alpha)
        quantizer = ModelQuantizer(model_path, config)
        quantizer.quantize_all_weights()
        stats = quantizer.get_summary_statistics()
        
        results[alpha] = {
            'avg_error': stats['avg_quantization_error'],
            'sparsity': stats['overall_sparsity'],
        }
        
        logger.info(f"  Alpha={alpha}: error={stats['avg_quantization_error']:.6f}, "
                   f"sparsity={stats['overall_sparsity']:.1%}")
    
    # Find optimal (minimize error while maintaining reasonable sparsity)
    best_alpha = min(results.keys(), key=lambda a: results[a]['avg_error'])
    
    return {
        'optimal_alpha': best_alpha,
        'results': results
    }


class GradientBasedAlphaOptimizer:
    """
    Gradient-based optimization of alpha parameter for each layer.
    
    Uses differentiable approximation of ternary quantization to find
    optimal alpha values that minimize reconstruction error.
    
    Theory:
        We use a straight-through estimator (STE) approximation where:
        - Forward: hard ternary quantization
        - Backward: gradient flows through soft approximation
        
        The soft ternary function uses tanh:
        q_soft(w) = tanh(β * (w - τ)) - tanh(β * (w + τ))
        
        where τ = α * mean(|w|) and β controls sharpness.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 num_iterations: int = 100,
                 beta: float = 10.0,
                 mse_weight: float = 1.0,
                 cosine_weight: float = 0.5):
        """
        Initialize the gradient-based optimizer.
        
        Args:
            learning_rate: Learning rate for alpha optimization
            num_iterations: Number of optimization iterations
            beta: Sharpness parameter for soft quantization
            mse_weight: Weight for MSE loss
            cosine_weight: Weight for cosine similarity loss
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = beta
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
    
    def soft_ternary_quantize(self, weights: torch.Tensor, 
                               alpha: torch.Tensor) -> torch.Tensor:
        """
        Soft (differentiable) ternary quantization.
        
        Args:
            weights: Weight tensor
            alpha: Threshold parameter (trainable)
            
        Returns:
            Soft-quantized weights
        """
        # Compute threshold
        abs_mean = torch.mean(torch.abs(weights))
        threshold = alpha * abs_mean
        
        # Soft ternary: approximates sign(w) * (|w| > threshold)
        # Using tanh for smooth approximation
        positive = torch.tanh(self.beta * (weights - threshold))
        negative = torch.tanh(self.beta * (-weights - threshold))
        
        soft_quantized = (positive - negative) / 2
        
        return soft_quantized
    
    def compute_loss(self, original: torch.Tensor, 
                     quantized: torch.Tensor,
                     scale: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss for optimization.
        
        Args:
            original: Original weights
            quantized: Soft-quantized weights
            scale: Scale factor
            
        Returns:
            Combined loss value
        """
        # Dequantize
        dequantized = quantized * scale
        
        # MSE loss
        mse_loss = torch.mean((original - dequantized) ** 2)
        
        # Cosine similarity loss
        orig_flat = original.flatten()
        deq_flat = dequantized.flatten()
        
        cosine_sim = torch.nn.functional.cosine_similarity(
            orig_flat.unsqueeze(0), 
            deq_flat.unsqueeze(0)
        )
        cosine_loss = 1 - cosine_sim
        
        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
        
        return total_loss
    
    def optimize_alpha(self, weights: np.ndarray, 
                       initial_alpha: float = 0.7,
                       verbose: bool = False) -> Tuple[float, Dict[str, Any]]:
        """
        Optimize alpha for a single layer using gradient descent.
        
        Args:
            weights: Original FP32 weights
            initial_alpha: Starting alpha value
            verbose: Print optimization progress
            
        Returns:
            Tuple of (optimal_alpha, optimization_stats)
        """
        # Convert to tensor
        weights_tensor = torch.from_numpy(weights.astype(np.float32))
        weights_tensor.requires_grad_(False)
        
        # Initialize alpha as trainable parameter
        alpha = torch.tensor([initial_alpha], requires_grad=True)
        
        # Optimizer
        optimizer = torch.optim.Adam([alpha], lr=self.learning_rate)
        
        # Track optimization
        loss_history = []
        alpha_history = []
        
        for i in range(self.num_iterations):
            optimizer.zero_grad()
            
            # Clamp alpha to valid range
            with torch.no_grad():
                alpha.clamp_(0.1, 0.99)
            
            # Soft quantize
            soft_quantized = self.soft_ternary_quantize(weights_tensor, alpha)
            
            # Compute scale (mean of non-zero magnitudes)
            # Use soft approximation: sigmoid-weighted mean
            importance = torch.sigmoid(self.beta * (torch.abs(soft_quantized) - 0.5))
            scale = torch.sum(importance * torch.abs(weights_tensor)) / (torch.sum(importance) + 1e-8)
            
            # Compute loss
            loss = self.compute_loss(weights_tensor, soft_quantized, scale)
            
            # Backward pass
            loss.backward()
            
            # Update alpha
            optimizer.step()
            
            # Record
            loss_history.append(loss.item())
            alpha_history.append(alpha.item())
            
            if verbose and (i + 1) % 20 == 0:
                logger.info(f"  Iter {i+1}: loss={loss.item():.6f}, alpha={alpha.item():.4f}")
        
        optimal_alpha = float(alpha.item())
        
        # Clamp to valid range
        optimal_alpha = max(0.1, min(0.99, optimal_alpha))
        
        stats = {
            'initial_alpha': initial_alpha,
            'final_alpha': optimal_alpha,
            'initial_loss': loss_history[0],
            'final_loss': loss_history[-1],
            'loss_reduction': (loss_history[0] - loss_history[-1]) / loss_history[0],
            'iterations': self.num_iterations,
        }
        
        return optimal_alpha, stats
    
    def optimize_model_alphas(self, model_path: str,
                               progress: bool = True) -> Dict[str, Tuple[float, Dict]]:
        """
        Optimize alpha for all layers in the model.
        
        Args:
            model_path: Path to model or HuggingFace ID
            progress: Show progress bar
            
        Returns:
            Dictionary mapping layer names to (optimal_alpha, stats)
        """
        try:
            from transformers import AutoModelForVision2Seq
            from tqdm import tqdm
        except ImportError:
            logger.error("transformers or tqdm not installed")
            return {}
        
        logger.info(f"Loading model: {model_path}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        
        results = {}
        
        # Get layers to optimize
        layers_to_optimize = []
        for name, param in model.named_parameters():
            if 'weight' not in name or param.ndim < 2:
                continue
            if param.numel() < 64:
                continue
            if any(x in name.lower() for x in ['layernorm', 'layer_norm', 'rmsnorm']):
                continue
            layers_to_optimize.append((name, param))
        
        logger.info(f"Optimizing alpha for {len(layers_to_optimize)} layers...")
        
        if progress:
            from tqdm import tqdm
            iterator = tqdm(layers_to_optimize, desc="Optimizing")
        else:
            iterator = layers_to_optimize
        
        for name, param in iterator:
            weights = param.detach().float().cpu().numpy()
            optimal_alpha, stats = self.optimize_alpha(weights)
            results[name] = (optimal_alpha, stats)
        
        return results
    
    def print_optimization_summary(self, results: Dict[str, Tuple[float, Dict]]) -> None:
        """Print summary of alpha optimization results."""
        if not results:
            print("No optimization results to display")
            return
        
        print("\n" + "=" * 70)
        print("GRADIENT-BASED ALPHA OPTIMIZATION SUMMARY")
        print("=" * 70)
        
        alphas = [r[0] for r in results.values()]
        reductions = [r[1]['loss_reduction'] for r in results.values()]
        
        print(f"\nLayers optimized: {len(results)}")
        print(f"\nAlpha distribution:")
        print(f"  Mean: {np.mean(alphas):.3f}")
        print(f"  Std:  {np.std(alphas):.3f}")
        print(f"  Min:  {np.min(alphas):.3f}")
        print(f"  Max:  {np.max(alphas):.3f}")
        
        print(f"\nLoss reduction:")
        print(f"  Mean: {np.mean(reductions):.1%}")
        print(f"  Max:  {np.max(reductions):.1%}")
        
        # Show layers with significant changes
        significant_changes = [
            (name, alpha, stats) 
            for name, (alpha, stats) in results.items()
            if abs(alpha - 0.7) > 0.1
        ]
        
        if significant_changes:
            print(f"\nLayers with significant alpha changes (>0.1 from 0.7):")
            for name, alpha, stats in sorted(significant_changes, key=lambda x: abs(x[1] - 0.7), reverse=True)[:10]:
                print(f"  {name}: α={alpha:.3f} (Δ={alpha-0.7:+.3f})")


def optimize_alpha_gradient(model_path: str,
                            learning_rate: float = 0.01,
                            num_iterations: int = 100,
                            progress: bool = True) -> Dict[str, Any]:
    """
    Convenience function for gradient-based alpha optimization.
    
    Args:
        model_path: Path to model or HuggingFace ID
        learning_rate: Learning rate for optimization
        num_iterations: Number of optimization iterations
        progress: Show progress bar
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = GradientBasedAlphaOptimizer(
        learning_rate=learning_rate,
        num_iterations=num_iterations
    )
    
    results = optimizer.optimize_model_alphas(model_path, progress=progress)
    optimizer.print_optimization_summary(results)
    
    # Create per-layer alpha configuration
    layer_alphas = {name: alpha for name, (alpha, _) in results.items()}
    
    return {
        'layer_alphas': layer_alphas,
        'detailed_results': results,
        'summary': {
            'mean_alpha': np.mean(list(layer_alphas.values())),
            'std_alpha': np.std(list(layer_alphas.values())),
            'num_layers': len(layer_alphas),
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ternary quantization for SmolVLM-256M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic quantization with default settings
    python quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    
    # Custom alpha and export
    python quantize_ternary.py --model ./model/smolvlm-256m --alpha 0.7 --output ./quantized
    
    # Per-channel quantization
    python quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct --mode per_channel
    
    # Search for optimal alpha
    python quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct --search-alpha
    
    # Gradient-based alpha optimization (recommended for best accuracy)
    python quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct --optimize-alpha
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Threshold factor (default: 0.7)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['per_tensor', 'per_channel', 'per_group'],
        default='per_tensor',
        help="Quantization mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./model/weights/quantized",
        help="Output directory"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export quantized weights"
    )
    parser.add_argument(
        "--search-alpha",
        action="store_true",
        help="Search for optimal alpha value"
    )
    parser.add_argument(
        "--optimize-alpha",
        action="store_true",
        help="Use gradient-based alpha optimization (per-layer)"
    )
    parser.add_argument(
        "--opt-lr",
        type=float,
        default=0.01,
        help="Learning rate for gradient-based optimization"
    )
    parser.add_argument(
        "--opt-iters",
        type=int,
        default=100,
        help="Number of iterations for gradient-based optimization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SiLens Ternary Quantizer")
    print("=" * 70)
    
    if args.search_alpha:
        results = search_optimal_alpha(args.model)
        print(f"\nOptimal alpha: {results['optimal_alpha']}")
        return
    
    if args.optimize_alpha:
        print("\nRunning gradient-based alpha optimization...")
        results = optimize_alpha_gradient(
            args.model,
            learning_rate=args.opt_lr,
            num_iterations=args.opt_iters,
            progress=True
        )
        
        # Save optimized alphas
        if args.export:
            import json
            from pathlib import Path
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / 'optimized_alphas.json', 'w') as f:
                json.dump({
                    'layer_alphas': results['layer_alphas'],
                    'summary': results['summary']
                }, f, indent=2)
            
            print(f"\nOptimized alphas saved to: {output_path / 'optimized_alphas.json'}")
        return
    
    # Create configuration
    mode_map = {
        'per_tensor': QuantizationMode.PER_TENSOR,
        'per_channel': QuantizationMode.PER_CHANNEL,
        'per_group': QuantizationMode.PER_GROUP,
    }
    
    config = TernaryQuantizationConfig(
        alpha=args.alpha,
        mode=mode_map[args.mode],
    )
    
    # Quantize
    quantizer = ModelQuantizer(args.model, config, device=args.device)
    quantizer.quantize_all_weights()
    quantizer.print_summary()
    
    # Export if requested
    if args.export:
        quantizer.export(args.output)
    
    print("\n" + "=" * 70)
    print("Quantization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
