"""
SiLens Model Configuration.

Handles loading model parameters and configuring hardware registers
for the vision encoder, projector, and language model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import struct

logger = logging.getLogger(__name__)


@dataclass
class VisionEncoderConfig:
    """Configuration for the SigLIP-B/16 vision encoder."""
    name: str = "siglip-b-16"
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    patch_size: int = 16
    image_size: int = 384
    num_patches: int = 576  # (384/16)^2
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6
    
    @property
    def num_parameters(self) -> int:
        """Estimate total parameters."""
        # Patch embedding: 3 * patch_size^2 * hidden_dim
        patch_embed = 3 * self.patch_size * self.patch_size * self.hidden_dim
        # Per transformer layer
        per_layer = (
            4 * self.hidden_dim * self.hidden_dim +  # Q, K, V, O projections
            2 * self.hidden_dim +  # Layer norms
            int(8 * self.hidden_dim * self.hidden_dim * self.mlp_ratio)  # MLP
        )
        return patch_embed + self.num_layers * per_layer


@dataclass
class ProjectorConfig:
    """Configuration for the multimodal projector."""
    name: str = "linear-projector"
    input_dim: int = 768
    output_dim: int = 576
    
    @property
    def num_parameters(self) -> int:
        """Estimate total parameters."""
        return self.input_dim * self.output_dim + self.output_dim


@dataclass
class LanguageModelConfig:
    """Configuration for the SmolLM2-135M language model."""
    name: str = "smollm2-135m"
    hidden_dim: int = 576
    num_layers: int = 30
    num_heads: int = 9
    head_dim: int = 64  # 576 / 9
    vocab_size: int = 49152
    max_seq_length: int = 8192
    intermediate_size: int = 1536  # ~3x hidden_dim
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    
    @property
    def num_parameters(self) -> int:
        """Estimate total parameters."""
        # Embeddings
        embed = self.vocab_size * self.hidden_dim
        # Per transformer layer
        per_layer = (
            4 * self.hidden_dim * self.hidden_dim +  # Q, K, V, O
            2 * self.hidden_dim * self.intermediate_size +  # MLP up/down
            2 * self.hidden_dim  # Layer norms
        )
        # LM head
        lm_head = self.hidden_dim * self.vocab_size
        return embed + self.num_layers * per_layer + lm_head


@dataclass
class ModelConfig:
    """Complete model configuration for SiLens."""
    name: str = "silens-vlm"
    version: str = "1.0.0"
    
    # Component configs
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    language: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    
    # Quantization settings
    weight_bits: int = 2  # Ternary: -1, 0, +1
    activation_bits: int = 8
    
    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    image_token_id: int = 32000

    
    @property
    def total_parameters(self) -> int:
        """Total model parameters."""
        return (
            self.vision.num_parameters +
            self.projector.num_parameters +
            self.language.num_parameters
        )
    
    @property
    def total_parameters_str(self) -> str:
        """Human-readable parameter count."""
        total = self.total_parameters
        if total >= 1e9:
            return f"{total/1e9:.1f}B"
        elif total >= 1e6:
            return f"{total/1e6:.1f}M"
        else:
            return f"{total/1e3:.1f}K"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.write_text(self.to_json())
        logger.info(f"Saved model config to {path}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        """Load from dictionary."""
        vision = VisionEncoderConfig(**data.pop("vision", {}))
        projector = ProjectorConfig(**data.pop("projector", {}))
        language = LanguageModelConfig(**data.pop("language", {}))
        return cls(vision=vision, projector=projector, language=language, **data)
    
    @classmethod
    def from_json(cls, json_str: str) -> ModelConfig:
        """Load from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def load(cls, path: Path) -> ModelConfig:
        """Load configuration from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return cls.from_json(path.read_text())


def load_model_config(path: Optional[Path] = None) -> ModelConfig:
    """
    Load model configuration.
    
    Args:
        path: Path to config file. If None, returns default config.
        
    Returns:
        ModelConfig instance
    """
    if path is not None:
        return ModelConfig.load(path)
    return ModelConfig()



class KVCacheManager:
    """
    Manages the KV cache for autoregressive generation.
    
    The KV cache stores key and value tensors from previous tokens
    to avoid recomputation during generation.
    """
    
    def __init__(self, config: LanguageModelConfig, max_batch_size: int = 1):
        self.config = config
        self.max_batch_size = max_batch_size
        
        # Calculate cache size per layer
        # K and V each: batch * seq_len * num_heads * head_dim
        self.cache_size_per_layer = (
            2 *  # K and V
            max_batch_size *
            config.max_seq_length *
            config.num_heads *
            config.head_dim *
            1  # 1 byte per element (INT8)
        )
        
        self.total_cache_size = self.cache_size_per_layer * config.num_layers
        
        # Cache state
        self.current_seq_len = 0
        self._cache_data: Optional[bytes] = None
    
    def reset(self) -> None:
        """Reset the cache for a new sequence."""
        self.current_seq_len = 0
        if self._cache_data:
            # Zero the cache
            self._cache_data = bytes(self.total_cache_size)
    
    def allocate(self) -> bytes:
        """Allocate cache memory."""
        self._cache_data = bytes(self.total_cache_size)
        return self._cache_data
    
    def get_cache_offset(self, layer_idx: int, is_key: bool) -> int:
        """Get byte offset for a layer's K or V cache."""
        layer_offset = layer_idx * self.cache_size_per_layer
        kv_offset = 0 if is_key else (self.cache_size_per_layer // 2)
        return layer_offset + kv_offset
    
    def update_seq_len(self, new_len: int) -> None:
        """Update current sequence length."""
        if new_len > self.config.max_seq_length:
            raise ValueError(f"Sequence length {new_len} exceeds max {self.config.max_seq_length}")
        self.current_seq_len = new_len


class HardwareConfigWriter:
    """
    Writes model configuration to hardware registers.
    
    Configures the hardware accelerator with model parameters
    before running inference.
    """
    
    # Register offsets for model configuration
    REG_VISION_CONFIG = 0x300
    REG_PROJ_CONFIG = 0x320
    REG_LLM_CONFIG = 0x340
    REG_QUANT_CONFIG = 0x360
    
    def __init__(self, device):
        self.device = device
    
    def write_config(self, config: ModelConfig) -> None:
        """Write complete model configuration to hardware."""
        self._write_vision_config(config.vision)
        self._write_projector_config(config.projector)
        self._write_llm_config(config.language)
        self._write_quant_config(config)
        logger.info("Model configuration written to hardware")

    
    def _write_vision_config(self, config: VisionEncoderConfig) -> None:
        """Write vision encoder configuration."""
        # Pack configuration into 32-bit registers
        # Reg 0: hidden_dim | num_layers
        self.device.write_reg(
            self.REG_VISION_CONFIG + 0,
            (config.hidden_dim << 16) | config.num_layers
        )
        # Reg 1: num_heads | patch_size
        self.device.write_reg(
            self.REG_VISION_CONFIG + 4,
            (config.num_heads << 16) | config.patch_size
        )
        # Reg 2: image_size | num_patches
        self.device.write_reg(
            self.REG_VISION_CONFIG + 8,
            (config.image_size << 16) | config.num_patches
        )
    
    def _write_projector_config(self, config: ProjectorConfig) -> None:
        """Write projector configuration."""
        self.device.write_reg(
            self.REG_PROJ_CONFIG,
            (config.input_dim << 16) | config.output_dim
        )
    
    def _write_llm_config(self, config: LanguageModelConfig) -> None:
        """Write language model configuration."""
        # Reg 0: hidden_dim | num_layers
        self.device.write_reg(
            self.REG_LLM_CONFIG + 0,
            (config.hidden_dim << 16) | config.num_layers
        )
        # Reg 1: num_heads | head_dim
        self.device.write_reg(
            self.REG_LLM_CONFIG + 4,
            (config.num_heads << 16) | config.head_dim
        )
        # Reg 2: vocab_size
        self.device.write_reg(
            self.REG_LLM_CONFIG + 8,
            config.vocab_size
        )
        # Reg 3: max_seq_length | intermediate_size
        self.device.write_reg(
            self.REG_LLM_CONFIG + 12,
            (config.max_seq_length << 16) | (config.intermediate_size & 0xFFFF)
        )
    
    def _write_quant_config(self, config: ModelConfig) -> None:
        """Write quantization configuration."""
        self.device.write_reg(
            self.REG_QUANT_CONFIG,
            (config.weight_bits << 8) | config.activation_bits
        )
