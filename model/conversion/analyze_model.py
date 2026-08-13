#!/usr/bin/env python3
"""
Analyze SmolVLM-256M model architecture and weights.

This script provides detailed information about:
- Model architecture (layers, dimensions)
- Weight statistics (mean, std, distribution)
- Quantization-friendliness analysis
- Memory requirements

Usage:
    python analyze_model.py --model model/smolvlm-256m
    python analyze_model.py --model HuggingFaceTB/SmolVLM-256M-Instruct
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np


def load_model(model_path: str):
    """Load model from local path or Hugging Face Hub."""
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError:
        print("ERROR: transformers not installed")
        print("Run: pip install transformers torch")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    return model


def analyze_architecture(model):
    """Analyze model architecture."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    
    # Count parameters by component
    component_params = defaultdict(int)
    
    for name, param in model.named_parameters():
        parts = name.split('.')
        if 'vision' in name or 'image' in name:
            component = 'vision_encoder'
        elif 'language' in name or 'model.layers' in name:
            component = 'language_model'
        elif 'multi_modal_projector' in name or 'projector' in name:
            component = 'projector'
        elif 'embed' in name:
            component = 'embeddings'
        elif 'lm_head' in name:
            component = 'lm_head'
        else:
            component = 'other'
        
        component_params[component] += param.numel()
    
    total_params = sum(component_params.values())
    
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print("\nBy component:")
    for comp, count in sorted(component_params.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_params
        print(f"  {comp:20s}: {count:>12,} ({pct:5.1f}%)")
    
    return component_params


def analyze_weights(model):
    """Analyze weight statistics."""
    print("\n" + "=" * 60)
    print("WEIGHT STATISTICS")
    print("=" * 60)
    
    stats = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data.float().cpu().numpy().flatten()
            
            stat = {
                'name': name,
                'shape': list(param.shape),
                'numel': param.numel(),
                'dtype': str(param.dtype),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'abs_mean': float(np.mean(np.abs(data))),
                'sparsity': float(np.mean(np.abs(data) < 0.01)),  # ~zero weights
            }
            stats.append(stat)
    
    # Print summary
    print(f"\nTotal layers: {len(stats)}")
    
    # Group by type
    weight_types = defaultdict(list)
    for s in stats:
        if 'weight' in s['name'] and 'norm' not in s['name']:
            weight_types['linear_weights'].append(s)
        elif 'bias' in s['name']:
            weight_types['biases'].append(s)
        elif 'norm' in s['name']:
            weight_types['norm_params'].append(s)
        elif 'embed' in s['name']:
            weight_types['embeddings'].append(s)
        else:
            weight_types['other'].append(s)
    
    for wtype, layers in weight_types.items():
        if not layers:
            continue
        
        print(f"\n{wtype.upper()}:")
        total_params = sum(l['numel'] for l in layers)
        avg_std = np.mean([l['std'] for l in layers])
        avg_sparsity = np.mean([l['sparsity'] for l in layers])
        
        print(f"  Count: {len(layers)}")
        print(f"  Total params: {total_params:,}")
        print(f"  Avg std: {avg_std:.4f}")
        print(f"  Avg sparsity (<0.01): {avg_sparsity:.1%}")
    
    return stats


def analyze_quantization_friendliness(model):
    """Analyze how well the model will quantize."""
    print("\n" + "=" * 60)
    print("QUANTIZATION ANALYSIS")
    print("=" * 60)
    
    results = {
        'good': [],
        'moderate': [],
        'challenging': []
    }
    
    for name, param in model.named_parameters():
        if 'weight' not in name or 'norm' in name:
            continue
        
        data = param.data.float().cpu().numpy().flatten()
        
        # Metrics for quantization friendliness
        std = np.std(data)
        kurtosis = float(((data - np.mean(data)) / std) ** 4).mean() - 3  # Excess kurtosis
        outlier_ratio = np.mean(np.abs(data) > 3 * std)
        
        # Score
        if kurtosis < 3 and outlier_ratio < 0.01:
            results['good'].append(name)
        elif kurtosis < 10 and outlier_ratio < 0.05:
            results['moderate'].append(name)
        else:
            results['challenging'].append(name)
    
    print(f"\nQuantization friendliness:")
    print(f"  Good (easy to quantize): {len(results['good'])} layers")
    print(f"  Moderate: {len(results['moderate'])} layers")
    print(f"  Challenging (may need special handling): {len(results['challenging'])} layers")
    
    if results['challenging']:
        print(f"\nChallenging layers:")
        for name in results['challenging'][:10]:
            print(f"    - {name}")
        if len(results['challenging']) > 10:
            print(f"    ... and {len(results['challenging']) - 10} more")
    
    return results


def estimate_hardware_requirements(model):
    """Estimate hardware requirements for hardwired implementation."""
    print("\n" + "=" * 60)
    print("HARDWARE REQUIREMENTS")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # For ternary weights: 2 bits per weight (but can be encoded as wire)
    # Estimation: ~6 transistors per weight for routing
    transistors_per_weight = 6
    total_transistors = total_params * transistors_per_weight
    
    # SKY130 density: ~500K transistors/mm² (conservative for routing-heavy)
    density = 500_000
    estimated_area = total_transistors / density
    
    # Memory for activations (8-bit)
    max_activation_size = 576 * 768  # Vision encoder output
    activation_memory = max_activation_size * 1  # 8-bit = 1 byte
    
    # KV cache for language model
    kv_cache_per_token = 30 * 2 * 576 * 1  # 30 layers, K+V, 576 dim, 8-bit
    kv_cache_2k = kv_cache_per_token * 2048
    
    print(f"\nWeight encoding:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Estimated transistors: {total_transistors:,}")
    print(f"  Estimated die area: {estimated_area:.0f} mm²")
    
    print(f"\nActivation memory:")
    print(f"  Max activation buffer: {activation_memory / 1024:.1f} KB")
    print(f"  KV cache (2K tokens): {kv_cache_2k / (1024*1024):.1f} MB")
    
    print(f"\nSKY130 feasibility:")
    max_reticle = 800  # mm²
    if estimated_area < max_reticle:
        print(f"  ✓ Fits in single reticle ({estimated_area:.0f} < {max_reticle} mm²)")
    else:
        print(f"  ✗ Exceeds reticle ({estimated_area:.0f} > {max_reticle} mm²)")
        print(f"    Consider: model pruning, multi-chip, or smaller model")
    
    return {
        'total_params': total_params,
        'estimated_area_mm2': estimated_area,
        'activation_memory_kb': activation_memory / 1024,
        'kv_cache_mb': kv_cache_2k / (1024 * 1024)
    }


def export_layer_info(model, output_path: str):
    """Export layer information to JSON for conversion tools."""
    layers = []
    
    for name, param in model.named_parameters():
        layers.append({
            'name': name,
            'shape': list(param.shape),
            'numel': param.numel(),
            'dtype': str(param.dtype),
            'requires_grad': param.requires_grad
        })
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(layers, f, indent=2)
    
    print(f"\nLayer info exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SmolVLM-256M model for SiLens conversion"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/smolvlm-256m",
        help="Path to model or Hugging Face model ID"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export layer info to JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-layer statistics"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SiLens Model Analyzer")
    print("=" * 60)
    
    # Load model
    model = load_model(args.model)
    
    # Run analyses
    arch_stats = analyze_architecture(model)
    weight_stats = analyze_weights(model)
    quant_results = analyze_quantization_friendliness(model)
    hw_reqs = estimate_hardware_requirements(model)
    
    # Export if requested
    if args.export:
        export_layer_info(model, args.export)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Model: SmolVLM-256M
Total parameters: {sum(arch_stats.values()):,}
Estimated die area: {hw_reqs['estimated_area_mm2']:.0f} mm²
Quantization readiness: {len(quant_results['good'])} good, {len(quant_results['challenging'])} challenging layers

Next steps:
1. Run quantization: python quantize.py
2. Validate accuracy: python validate.py
3. Generate Verilog: python convert_to_verilog.py
""")


if __name__ == "__main__":
    main()
