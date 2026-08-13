"""
SiLens Tokenizer Module.

Text tokenization wrapper for SmolLM2 language model with support for:
- HuggingFace tokenizers integration
- Vision-language special tokens
- Chat template formatting
- Fallback tokenization when transformers unavailable
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    PreTrainedTokenizer = None
    HAS_TRANSFORMERS = False


# Special tokens for SiLens VLM
SPECIAL_TOKENS = {
    "bos_token": "<|begin_of_text|>",
    "eos_token": "<|end_of_text|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
    "image_token": "<image>",
    "image_start": "<|image_start|>",
    "image_end": "<|image_end|>",
}

# Default SmolLM2 tokenizer
DEFAULT_TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_VOCAB_SIZE = 49152


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""
    name: str = DEFAULT_TOKENIZER_NAME
    vocab_size: int = DEFAULT_VOCAB_SIZE
    max_length: int = 8192
    padding_side: str = "right"
    truncation_side: str = "right"
    add_special_tokens: bool = True
    trust_remote_code: bool = True


class SiLensTokenizer:
    """
    Tokenizer wrapper for SiLens VLM.
    
    Provides a unified interface for text tokenization that works with
    HuggingFace tokenizers when available, with a fallback implementation
    for environments without transformers installed.
    
    Example:
        tokenizer = SiLensTokenizer()
        
        # Encode text
        tokens = tokenizer.encode("Describe this image.")
        
        # Decode tokens
        text = tokenizer.decode(tokens)
        
        # Format VLM prompt
        prompt = tokenizer.format_vlm_prompt(
            "What objects are visible?",
            num_image_tokens=576
        )
    """
    
    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize tokenizer.
        
        Args:
            config: Tokenizer configuration
            tokenizer_path: Optional path to local tokenizer files
        """
        self.config = config or TokenizerConfig()
        self._tokenizer = None
        self._is_fallback = False
        
        self._init_tokenizer(tokenizer_path)

    
    def _init_tokenizer(self, tokenizer_path: Optional[Union[str, Path]] = None):
        """Initialize the underlying tokenizer."""
        if HAS_TRANSFORMERS:
            try:
                name_or_path = str(tokenizer_path) if tokenizer_path else self.config.name
                self._tokenizer = AutoTokenizer.from_pretrained(
                    name_or_path,
                    trust_remote_code=self.config.trust_remote_code,
                    padding_side=self.config.padding_side,
                    truncation_side=self.config.truncation_side,
                )
                # Add special tokens if not present
                self._add_special_tokens()
                logger.info(f"Loaded tokenizer: {name_or_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        
        # Use fallback tokenizer
        logger.info("Using fallback tokenizer")
        self._tokenizer = FallbackTokenizer(self.config.vocab_size)
        self._is_fallback = True
    
    def _add_special_tokens(self):
        """Add VLM special tokens to the tokenizer."""
        if self._is_fallback:
            return
        
        special_tokens_to_add = []
        for key, token in SPECIAL_TOKENS.items():
            if not hasattr(self._tokenizer, key) or getattr(self._tokenizer, key) is None:
                special_tokens_to_add.append(token)
        
        if special_tokens_to_add:
            self._tokenizer.add_special_tokens({
                "additional_special_tokens": special_tokens_to_add
            })
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if hasattr(self._tokenizer, "vocab_size"):
            return self._tokenizer.vocab_size
        return self.config.vocab_size
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        if hasattr(self._tokenizer, "bos_token_id") and self._tokenizer.bos_token_id:
            return self._tokenizer.bos_token_id
        return 1

    
    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        if hasattr(self._tokenizer, "eos_token_id") and self._tokenizer.eos_token_id:
            return self._tokenizer.eos_token_id
        return 2
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        if hasattr(self._tokenizer, "pad_token_id") and self._tokenizer.pad_token_id:
            return self._tokenizer.pad_token_id
        return 0
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            truncation: Whether to truncate long sequences
            padding: Whether to pad short sequences
            
        Returns:
            List of token IDs
        """
        max_len = max_length or self.config.max_length
        
        if self._is_fallback:
            tokens = self._tokenizer.encode(text)
            if add_special_tokens:
                tokens = [self.bos_token_id] + tokens
            if truncation and len(tokens) > max_len:
                tokens = tokens[:max_len]
            if padding and len(tokens) < max_len:
                tokens = tokens + [self.pad_token_id] * (max_len - len(tokens))
            return tokens
        
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_len,
            truncation=truncation,
            padding="max_length" if padding else False,
        )
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            clean_up_tokenization_spaces: Whether to clean up extra spaces
            
        Returns:
            Decoded text string
        """
        if self._is_fallback:
            return self._tokenizer.decode(token_ids, skip_special_tokens)
        
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            padding: Whether to pad to same length
            
        Returns:
            List of token ID lists
        """
        return [
            self.encode(text, add_special_tokens, max_length, padding=padding)
            for text in texts
        ]
    
    def decode_batch(
        self,
        token_ids_list: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token sequences.
        
        Args:
            token_ids_list: List of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded texts
        """
        return [
            self.decode(ids, skip_special_tokens)
            for ids in token_ids_list
        ]
    
    def format_vlm_prompt(
        self,
        text_prompt: str,
        num_image_tokens: int = 576,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Format a vision-language model prompt.
        
        Args:
            text_prompt: User's text prompt
            num_image_tokens: Number of image tokens (patches)
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string with image placeholder
        """
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}\n\n")
        
        # Add image placeholder
        parts.append(SPECIAL_TOKENS["image_start"])
        parts.append(SPECIAL_TOKENS["image_token"] * num_image_tokens)
        parts.append(SPECIAL_TOKENS["image_end"])
        parts.append("\n\n")
        
        # Add user prompt
        parts.append(f"User: {text_prompt}\n\nAssistant:")
        
        return "".join(parts)

    
    def format_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format a chat conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add generation prompt at end
            
        Returns:
            Formatted chat string
        """
        if not self._is_fallback and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        
        # Fallback chat formatting
        formatted = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        
        result = "\n\n".join(formatted)
        if add_generation_prompt:
            result += "\n\nAssistant:"
        
        return result
    
    def get_token_info(self, token_id: int) -> Dict[str, Any]:
        """
        Get information about a specific token.
        
        Args:
            token_id: Token ID to look up
            
        Returns:
            Dict with token string and metadata
        """
        if self._is_fallback:
            return {
                "id": token_id,
                "token": chr(token_id) if 32 <= token_id < 127 else f"<{token_id}>",
                "is_special": token_id < 10,
            }
        
        token_str = self._tokenizer.convert_ids_to_tokens(token_id)
        return {
            "id": token_id,
            "token": token_str,
            "is_special": token_id in self._tokenizer.all_special_ids,
        }


class FallbackTokenizer:
    """Simple fallback tokenizer when transformers is not available."""
    
    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self._special_ids = {0, 1, 2}  # PAD, BOS, EOS
    
    def encode(self, text: str) -> List[int]:
        """Simple byte-level encoding."""
        # Use byte values but ensure they fit vocab
        tokens = []
        for char in text:
            byte_val = ord(char)
            # Map to vocab range, reserving first 10 for special tokens
            token_id = 10 + (byte_val % (self.vocab_size - 10))
            tokens.append(token_id)
        return tokens
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode tokens back to text."""
        chars = []
        for tid in token_ids:
            if skip_special and tid in self._special_ids:
                continue
            if tid >= 10:
                # Reverse the encoding
                char_code = (tid - 10) % 128
                if 32 <= char_code < 127:
                    chars.append(chr(char_code))
                else:
                    chars.append("?")
        return "".join(chars)


# Convenience functions

def get_tokenizer(
    name: Optional[str] = None,
    **kwargs,
) -> SiLensTokenizer:
    """
    Get a tokenizer instance.
    
    Args:
        name: Tokenizer name or path (uses default if None)
        **kwargs: Additional config options
        
    Returns:
        SiLensTokenizer instance
    """
    config = TokenizerConfig(name=name or DEFAULT_TOKENIZER_NAME, **kwargs)
    return SiLensTokenizer(config)


def count_tokens(text: str, tokenizer: Optional[SiLensTokenizer] = None) -> int:
    """
    Count tokens in a text string.
    
    Args:
        text: Input text
        tokenizer: Optional tokenizer (creates default if None)
        
    Returns:
        Number of tokens
    """
    if tokenizer is None:
        tokenizer = SiLensTokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))
