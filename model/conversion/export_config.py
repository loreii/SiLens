#!/usr/bin/env python3
"""
Export model configuration for SiLens hardware and SDK.

This module exports the complete SmolVLM-256M architecture configuration
in multiple formats:
- JSON config for SDK/driver integration
- Verilog parameter files for RTL synthesis
- C header files for firmware
- Python config for validation tools

Usage:
    python export_config.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    python export_config.py --model HuggingFaceTB/SmolVLM-256M-Instruct --output ./config

Author: SiLens AI/ML Team
License: Apache 2.0
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VisionEncoderConfig:
    """Configuration for the vision encoder (SigLIP)."""
    model_type: str = "siglip"
    image_size: int = 384
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    attention_head_dim: int = 72
    num_patches: int = 729  # (384/14)^2 = 27^2 - 1 for cls + patches
    
    # Derived values
    @property
    def patches_per_side(self) -> int:
        return self.image_size // self.patch_size
    
    @property
    def total_patches(self) -> int:
        return self.patches_per_side ** 2
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['patches_per_side'] = self.patches_per_side
        d['total_patches'] = self.total_patches
        return d


@dataclass
class ProjectorConfig:
    """Configuration for the multimodal projector."""
    input_hidden_size: int = 1152   # From vision encoder
    output_hidden_size: int = 576   # To language model
    num_layers: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LanguageModelConfig:
    """Configuration for the language model (SmolLM)."""
    model_type: str = "llama"
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    head_dim: int = 64
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    
    # Derived values
    @property
    def kv_channels(self) -> int:
        return self.head_dim * self.num_key_value_heads
    
    @property
    def qkv_size(self) -> int:
        return self.hidden_size + 2 * self.kv_channels
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['kv_channels'] = self.kv_channels
        d['qkv_size'] = self.qkv_size
        return d


@dataclass
class HardwareConfig:
    """Hardware-specific configuration for RTL."""
    # Bit widths
    activation_bits: int = 8
    accumulator_bits: int = 32
    weight_bits: int = 2  # Ternary encoding
    
    # Memory configuration
    sram_kv_cache_kb: int = 512
    sram_activation_kb: int = 64
    
    # Processing configuration
    parallel_mac_units: int = 64
    clock_freq_mhz: int = 100
    
    # Interface
    axi_data_width: int = 64
    axi_addr_width: int = 32
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SmolVLMConfig:
    """Complete SmolVLM-256M configuration."""
    model_name: str = "SmolVLM-256M"
    model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    
    vision_encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # Computed properties
    @property
    def total_parameters(self) -> int:
        """Estimate total parameters."""
        # Vision encoder
        ve = self.vision_encoder
        ve_params = (
            ve.num_hidden_layers * (
                4 * ve.hidden_size * ve.hidden_size +  # QKV + output
                2 * ve.hidden_size * ve.intermediate_size  # MLP
            ) +
            ve.hidden_size * 3 * ve.patch_size * ve.patch_size  # Patch embedding
        )
        
        # Projector
        proj_params = (
            self.projector.input_hidden_size * self.projector.output_hidden_size +
            self.projector.output_hidden_size * self.projector.output_hidden_size
        )
        
        # Language model
        lm = self.language_model
        lm_params = (
            lm.vocab_size * lm.hidden_size +  # Embeddings
            lm.num_hidden_layers * (
                lm.hidden_size * (lm.hidden_size + 2 * lm.kv_channels) +  # Attention
                lm.hidden_size * lm.hidden_size +  # Output proj
                2 * lm.hidden_size * lm.intermediate_size  # MLP
            ) +
            lm.vocab_size * lm.hidden_size  # LM head
        )
        
        return ve_params + proj_params + lm_params
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'model_id': self.model_id,
            'total_parameters_estimate': self.total_parameters,
            'vision_encoder': self.vision_encoder.to_dict(),
            'projector': self.projector.to_dict(),
            'language_model': self.language_model.to_dict(),
            'hardware': self.hardware.to_dict(),
        }


class ConfigExporter:
    """
    Export model configuration to various formats.
    
    Supports:
    - JSON for SDK/driver
    - Verilog parameters for RTL
    - C headers for firmware
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the exporter.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to load model on
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.config = SmolVLMConfig()
        self.model_config = None
    
    def load_model_config(self) -> None:
        """Load configuration from the actual model."""
        try:
            from transformers import AutoConfig
        except ImportError:
            logger.error("transformers not installed")
            sys.exit(1)
        
        logger.info(f"Loading config from: {self.model_path}")
        
        try:
            full_config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model_config = full_config
            
            # Update our config from the actual model
            self._update_from_model_config(full_config)
            
        except Exception as e:
            logger.warning(f"Could not load model config: {e}")
            logger.info("Using default SmolVLM-256M configuration")
    
    def _update_from_model_config(self, model_config) -> None:
        """Update internal config from model's config."""
        # Vision encoder config
        if hasattr(model_config, 'vision_config'):
            vc = model_config.vision_config
            self.config.vision_encoder = VisionEncoderConfig(
                model_type=getattr(vc, 'model_type', 'siglip'),
                image_size=getattr(vc, 'image_size', 384),
                patch_size=getattr(vc, 'patch_size', 14),
                num_channels=getattr(vc, 'num_channels', 3),
                hidden_size=getattr(vc, 'hidden_size', 1152),
                intermediate_size=getattr(vc, 'intermediate_size', 4304),
                num_hidden_layers=getattr(vc, 'num_hidden_layers', 27),
                num_attention_heads=getattr(vc, 'num_attention_heads', 16),
                attention_head_dim=getattr(vc, 'hidden_size', 1152) // getattr(vc, 'num_attention_heads', 16),
            )

        
        # Language model config
        if hasattr(model_config, 'text_config'):
            tc = model_config.text_config
            self.config.language_model = LanguageModelConfig(
                model_type=getattr(tc, 'model_type', 'llama'),
                vocab_size=getattr(tc, 'vocab_size', 49152),
                hidden_size=getattr(tc, 'hidden_size', 576),
                intermediate_size=getattr(tc, 'intermediate_size', 1536),
                num_hidden_layers=getattr(tc, 'num_hidden_layers', 30),
                num_attention_heads=getattr(tc, 'num_attention_heads', 9),
                num_key_value_heads=getattr(tc, 'num_key_value_heads', 3),
                head_dim=getattr(tc, 'head_dim', 64),
                max_position_embeddings=getattr(tc, 'max_position_embeddings', 8192),
                rope_theta=getattr(tc, 'rope_theta', 10000.0),
                rms_norm_eps=getattr(tc, 'rms_norm_eps', 1e-5),
            )
        
        # Projector config - often inferred from dimensions
        if hasattr(model_config, 'vision_config') and hasattr(model_config, 'text_config'):
            self.config.projector = ProjectorConfig(
                input_hidden_size=model_config.vision_config.hidden_size,
                output_hidden_size=model_config.text_config.hidden_size,
            )
        
        logger.info("Configuration updated from model")
    
    def export_json_config(self, output_path: str) -> None:
        """
        Export configuration as JSON for SDK/driver.
        
        Args:
            output_path: Path to output JSON file
        """
        if self.model_config is None:
            self.load_model_config()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.config.to_dict()
        config_dict['exported_at'] = datetime.now().isoformat()
        config_dict['source_model'] = self.model_path
        
        with open(output_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"JSON config exported to: {output_file}")

    
    def export_verilog_params(self, output_path: str) -> None:
        """
        Export configuration as Verilog parameters.
        
        Args:
            output_path: Path to output Verilog header file
        """
        if self.model_config is None:
            self.load_model_config()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        ve = self.config.vision_encoder
        proj = self.config.projector
        lm = self.config.language_model
        hw = self.config.hardware
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        verilog = f"""// =============================================================================
// SiLens Model Parameters
// Auto-generated from {self.model_path}
// Generated: {timestamp}
// =============================================================================
// DO NOT EDIT - This file is auto-generated by export_config.py
// =============================================================================

`ifndef SILENS_MODEL_PARAMS_VH
`define SILENS_MODEL_PARAMS_VH

// =============================================================================
// Model Identification
// =============================================================================
// Model: SmolVLM-256M
// Total Parameters: ~{self.config.total_parameters:,}

// =============================================================================
// Vision Encoder Parameters (SigLIP)
// =============================================================================
parameter VE_IMAGE_SIZE        = {ve.image_size};
parameter VE_PATCH_SIZE        = {ve.patch_size};
parameter VE_NUM_CHANNELS      = {ve.num_channels};
parameter VE_HIDDEN_SIZE       = {ve.hidden_size};
parameter VE_INTERMEDIATE_SIZE = {ve.intermediate_size};
parameter VE_NUM_LAYERS        = {ve.num_hidden_layers};
parameter VE_NUM_HEADS         = {ve.num_attention_heads};
parameter VE_HEAD_DIM          = {ve.attention_head_dim};
parameter VE_PATCHES_PER_SIDE  = {ve.patches_per_side};
parameter VE_TOTAL_PATCHES     = {ve.total_patches};

// =============================================================================
// Projector Parameters
// =============================================================================
parameter PROJ_INPUT_DIM  = {proj.input_hidden_size};
parameter PROJ_OUTPUT_DIM = {proj.output_hidden_size};
parameter PROJ_NUM_LAYERS = {proj.num_layers};

// =============================================================================
// Language Model Parameters (SmolLM)
// =============================================================================
parameter LM_VOCAB_SIZE        = {lm.vocab_size};
parameter LM_HIDDEN_SIZE       = {lm.hidden_size};
parameter LM_INTERMEDIATE_SIZE = {lm.intermediate_size};
parameter LM_NUM_LAYERS        = {lm.num_hidden_layers};
parameter LM_NUM_HEADS         = {lm.num_attention_heads};
parameter LM_NUM_KV_HEADS      = {lm.num_key_value_heads};
parameter LM_HEAD_DIM          = {lm.head_dim};
parameter LM_MAX_SEQ_LEN       = {lm.max_position_embeddings};
parameter LM_KV_CHANNELS       = {lm.kv_channels};

// =============================================================================
// Hardware Configuration
// =============================================================================
parameter ACT_WIDTH     = {hw.activation_bits};
parameter ACC_WIDTH     = {hw.accumulator_bits};
parameter WEIGHT_BITS   = {hw.weight_bits};
parameter PARALLEL_MACS = {hw.parallel_mac_units};

// Memory (in bytes)
parameter SRAM_KV_CACHE_SIZE   = {hw.sram_kv_cache_kb * 1024};
parameter SRAM_ACTIVATION_SIZE = {hw.sram_activation_kb * 1024};

// Interface
parameter AXI_DATA_WIDTH = {hw.axi_data_width};
parameter AXI_ADDR_WIDTH = {hw.axi_addr_width};

// =============================================================================
// Derived Constants
// =============================================================================
// Vision encoder total parameters per layer
parameter VE_LAYER_PARAMS = VE_HIDDEN_SIZE * (4 * VE_HIDDEN_SIZE + 2 * VE_INTERMEDIATE_SIZE);

// Language model parameters per layer  
parameter LM_LAYER_PARAMS = LM_HIDDEN_SIZE * (LM_HIDDEN_SIZE + 2 * LM_KV_CHANNELS + LM_HIDDEN_SIZE + 2 * LM_INTERMEDIATE_SIZE);

// KV cache size per token (bytes)
parameter KV_CACHE_PER_TOKEN = LM_NUM_LAYERS * 2 * LM_KV_CHANNELS;

`endif // SILENS_MODEL_PARAMS_VH
"""
        
        with open(output_file, 'w') as f:
            f.write(verilog)
        
        logger.info(f"Verilog parameters exported to: {output_file}")

    
    def export_c_header(self, output_path: str) -> None:
        """
        Export configuration as C header for firmware.
        
        Args:
            output_path: Path to output C header file
        """
        if self.model_config is None:
            self.load_model_config()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        ve = self.config.vision_encoder
        proj = self.config.projector
        lm = self.config.language_model
        hw = self.config.hardware
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        header = f"""/**
 * SiLens Model Configuration
 * Auto-generated from {self.model_path}
 * Generated: {timestamp}
 * 
 * DO NOT EDIT - This file is auto-generated by export_config.py
 */

#ifndef SILENS_MODEL_CONFIG_H
#define SILENS_MODEL_CONFIG_H

#include <stdint.h>

/* ============================================================================
 * Vision Encoder Configuration (SigLIP)
 * ============================================================================ */
#define VE_IMAGE_SIZE        {ve.image_size}
#define VE_PATCH_SIZE        {ve.patch_size}
#define VE_NUM_CHANNELS      {ve.num_channels}
#define VE_HIDDEN_SIZE       {ve.hidden_size}
#define VE_INTERMEDIATE_SIZE {ve.intermediate_size}
#define VE_NUM_LAYERS        {ve.num_hidden_layers}
#define VE_NUM_HEADS         {ve.num_attention_heads}
#define VE_HEAD_DIM          {ve.attention_head_dim}
#define VE_PATCHES_PER_SIDE  {ve.patches_per_side}
#define VE_TOTAL_PATCHES     {ve.total_patches}

/* ============================================================================
 * Projector Configuration
 * ============================================================================ */
#define PROJ_INPUT_DIM  {proj.input_hidden_size}
#define PROJ_OUTPUT_DIM {proj.output_hidden_size}
#define PROJ_NUM_LAYERS {proj.num_layers}

/* ============================================================================
 * Language Model Configuration (SmolLM)
 * ============================================================================ */
#define LM_VOCAB_SIZE        {lm.vocab_size}
#define LM_HIDDEN_SIZE       {lm.hidden_size}
#define LM_INTERMEDIATE_SIZE {lm.intermediate_size}
#define LM_NUM_LAYERS        {lm.num_hidden_layers}
#define LM_NUM_HEADS         {lm.num_attention_heads}
#define LM_NUM_KV_HEADS      {lm.num_key_value_heads}
#define LM_HEAD_DIM          {lm.head_dim}
#define LM_MAX_SEQ_LEN       {lm.max_position_embeddings}
#define LM_KV_CHANNELS       {lm.kv_channels}

/* ============================================================================
 * Hardware Configuration
 * ============================================================================ */
#define ACT_WIDTH       {hw.activation_bits}
#define ACC_WIDTH       {hw.accumulator_bits}
#define WEIGHT_BITS     {hw.weight_bits}
#define PARALLEL_MACS   {hw.parallel_mac_units}

#define SRAM_KV_CACHE_SIZE   ({hw.sram_kv_cache_kb} * 1024)
#define SRAM_ACTIVATION_SIZE ({hw.sram_activation_kb} * 1024)

#define AXI_DATA_WIDTH {hw.axi_data_width}
#define AXI_ADDR_WIDTH {hw.axi_addr_width}

/* ============================================================================
 * Derived Constants
 * ============================================================================ */
#define KV_CACHE_PER_TOKEN (LM_NUM_LAYERS * 2 * LM_KV_CHANNELS)
#define MAX_KV_CACHE_TOKENS (SRAM_KV_CACHE_SIZE / KV_CACHE_PER_TOKEN)

/* ============================================================================
 * Configuration Structures
 * ============================================================================ */
typedef struct {{
    uint16_t image_size;
    uint16_t patch_size;
    uint16_t hidden_size;
    uint16_t num_layers;
    uint16_t num_heads;
}} silens_vision_config_t;

typedef struct {{
    uint16_t input_dim;
    uint16_t output_dim;
    uint8_t num_layers;
}} silens_projector_config_t;

typedef struct {{
    uint32_t vocab_size;
    uint16_t hidden_size;
    uint16_t num_layers;
    uint8_t num_heads;
    uint8_t num_kv_heads;
    uint16_t max_seq_len;
}} silens_lm_config_t;

typedef struct {{
    silens_vision_config_t vision;
    silens_projector_config_t projector;
    silens_lm_config_t language_model;
}} silens_model_config_t;

/* Global config instance (defined in silens_config.c) */
extern const silens_model_config_t SILENS_CONFIG;

#endif /* SILENS_MODEL_CONFIG_H */
"""
        
        with open(output_file, 'w') as f:
            f.write(header)
        
        logger.info(f"C header exported to: {output_file}")

    
    def export_layer_info(self, output_path: str) -> None:
        """
        Export detailed layer-by-layer information.
        
        Args:
            output_path: Path to output JSON file
        """
        if self.model is None:
            try:
                import torch
                from transformers import AutoModelForVision2Seq
                
                logger.info("Loading model for layer analysis...")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
                return
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        layers = []
        
        for name, param in self.model.named_parameters():
            layer_info = {
                'name': name,
                'shape': list(param.shape),
                'numel': param.numel(),
                'dtype': str(param.dtype),
                'requires_grad': param.requires_grad,
                'component': self._classify_component(name),
            }
            
            # Add layer type classification
            if 'weight' in name and 'norm' not in name:
                layer_info['type'] = 'linear'
            elif 'bias' in name:
                layer_info['type'] = 'bias'
            elif 'norm' in name:
                layer_info['type'] = 'normalization'
            elif 'embed' in name:
                layer_info['type'] = 'embedding'
            else:
                layer_info['type'] = 'other'
            
            layers.append(layer_info)
        
        output_data = {
            'model_path': self.model_path,
            'total_layers': len(layers),
            'total_parameters': sum(l['numel'] for l in layers),
            'layers': layers,
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Layer info exported to: {output_file}")

    
    def _classify_component(self, name: str) -> str:
        """Classify layer by component."""
        name_lower = name.lower()
        
        if any(x in name_lower for x in ['vision', 'image', 'siglip']):
            return 'vision_encoder'
        elif any(x in name_lower for x in ['projector', 'connector', 'multi_modal']):
            return 'projector'
        elif any(x in name_lower for x in ['language', 'model.layers', 'lm_head']):
            return 'language_model'
        elif 'embed' in name_lower:
            return 'embeddings'
        else:
            return 'other'
    
    def export_all(self, output_dir: str) -> None:
        """
        Export all configuration formats.
        
        Args:
            output_dir: Directory to save all config files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.export_json_config(str(output_path / 'model_config.json'))
        self.export_verilog_params(str(output_path / 'model_params.vh'))
        self.export_c_header(str(output_path / 'silens_model_config.h'))
        self.export_layer_info(str(output_path / 'layer_info.json'))
        
        logger.info(f"All configuration files exported to: {output_path}")
    
    def print_config(self) -> None:
        """Print configuration summary to console."""
        if self.model_config is None:
            self.load_model_config()
        
        print("\n" + "=" * 70)
        print("SmolVLM-256M Configuration")
        print("=" * 70)
        
        ve = self.config.vision_encoder
        print(f"\n--- Vision Encoder ({ve.model_type}) ---")
        print(f"  Image size: {ve.image_size}x{ve.image_size}")
        print(f"  Patch size: {ve.patch_size}x{ve.patch_size}")
        print(f"  Patches: {ve.patches_per_side}x{ve.patches_per_side} = {ve.total_patches}")
        print(f"  Hidden size: {ve.hidden_size}")
        print(f"  Layers: {ve.num_hidden_layers}")
        print(f"  Attention heads: {ve.num_attention_heads}")
        
        proj = self.config.projector
        print(f"\n--- Projector ---")
        print(f"  Input dim: {proj.input_hidden_size}")
        print(f"  Output dim: {proj.output_hidden_size}")
        print(f"  Layers: {proj.num_layers}")
        
        lm = self.config.language_model
        print(f"\n--- Language Model ({lm.model_type}) ---")
        print(f"  Vocab size: {lm.vocab_size:,}")
        print(f"  Hidden size: {lm.hidden_size}")
        print(f"  Layers: {lm.num_hidden_layers}")
        print(f"  Attention heads: {lm.num_attention_heads}")
        print(f"  KV heads: {lm.num_key_value_heads}")
        print(f"  Max sequence length: {lm.max_position_embeddings:,}")
        
        hw = self.config.hardware
        print(f"\n--- Hardware Config ---")
        print(f"  Activation bits: {hw.activation_bits}")
        print(f"  Weight bits: {hw.weight_bits} (ternary)")
        print(f"  Parallel MACs: {hw.parallel_mac_units}")
        print(f"  KV cache SRAM: {hw.sram_kv_cache_kb} KB")
        
        print(f"\n--- Summary ---")
        print(f"  Estimated parameters: {self.config.total_parameters:,}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export SmolVLM-256M configuration for SiLens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export all config formats
    python export_config.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    
    # Custom output directory
    python export_config.py --model HuggingFaceTB/SmolVLM-256M-Instruct --output ./config
    
    # Export only specific formats
    python export_config.py --model HuggingFaceTB/SmolVLM-256M-Instruct --format json
    python export_config.py --model HuggingFaceTB/SmolVLM-256M-Instruct --format verilog
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
        default="./model/config",
        help="Output directory for config files"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['all', 'json', 'verilog', 'c', 'layers'],
        default='all',
        help="Export format (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SiLens Configuration Exporter")
    print("=" * 70)
    
    exporter = ConfigExporter(args.model, device=args.device)
    exporter.load_model_config()
    
    # Print config summary
    exporter.print_config()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'all':
        exporter.export_all(str(output_path))
    elif args.format == 'json':
        exporter.export_json_config(str(output_path / 'model_config.json'))
    elif args.format == 'verilog':
        exporter.export_verilog_params(str(output_path / 'model_params.vh'))
    elif args.format == 'c':
        exporter.export_c_header(str(output_path / 'silens_model_config.h'))
    elif args.format == 'layers':
        exporter.export_layer_info(str(output_path / 'layer_info.json'))
    
    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
