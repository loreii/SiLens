"""
SiLens Inference Engine.

Provides high-level API for running vision-language inference on SiLens hardware.
Handles image preprocessing, tokenization, inference execution, and output decoding.
"""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Iterator, AsyncIterator, Union, List, Any
from pathlib import Path
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from silens.device import (
    SiLensDevice, 
    SimulatedDevice,
    Registers, 
    StatusBits,
    CtrlBits,
    DeviceError,
    DeviceTimeoutError,
    IMG_SIZE,
    VOCAB_SIZE,
)
from silens.model import ModelConfig, load_model_config
from silens.utils import load_image, preprocess_image, Timer

logger = logging.getLogger(__name__)


# Type alias for streaming callback
StreamingCallback = Callable[[str], None]


@dataclass
class InferenceResult:
    """Result of an inference operation."""
    text: str
    tokens: List[int]
    num_tokens: int
    vision_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    tokens_per_second: float
    
    # Optional detailed timing
    preprocessing_time_ms: float = 0.0
    tokenization_time_ms: float = 0.0
    
    # Model info
    model_name: str = ""
    
    def __str__(self) -> str:
        return self.text
    
    def summary(self) -> str:
        """Return a summary of inference statistics."""
        return (
            f"Output: {len(self.text)} chars, {self.num_tokens} tokens\n"
            f"Timing: {self.total_time_ms:.1f}ms total "
            f"(vision: {self.vision_time_ms:.1f}ms, "
            f"generation: {self.generation_time_ms:.1f}ms)\n"
            f"Speed: {self.tokens_per_second:.1f} tokens/sec"
        )


class InferenceEngine:
    """
    High-level inference engine for SiLens devices.
    
    Handles the complete inference pipeline:
    1. Image loading and preprocessing
    2. Tokenization of text prompts
    3. Hardware inference execution
    4. Output token decoding
    
    Example:
        device = SiLensDevice.discover()[0]
        engine = InferenceEngine(device)
        
        # Simple usage
        result = engine.describe_image("photo.jpg")
        print(result.text)
        
        # With custom prompt
        result = engine.run("photo.jpg", "What color is the car?")
        
        # Streaming output
        for token in engine.stream("photo.jpg", "Describe this image"):
            print(token, end="", flush=True)
    """

    
    # Default tokenizer for SmolLM2
    DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
    
    # Special tokens
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    PAD_TOKEN_ID = 0
    IMAGE_TOKEN = "<image>"
    
    def __init__(
        self,
        device: SiLensDevice,
        model_config: Optional[ModelConfig] = None,
        tokenizer_name: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        """
        Initialize inference engine.
        
        Args:
            device: SiLens device instance
            model_config: Model configuration (uses default if None)
            tokenizer_name: HuggingFace tokenizer name/path
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        """
        self.device = device
        self.model_config = model_config or load_model_config()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Initialize tokenizer
        self._tokenizer = None
        self._tokenizer_name = tokenizer_name or self.DEFAULT_TOKENIZER
        
        # Allocate buffers
        self._image_buffer = None
        self._output_buffer = None
        self._kv_cache_buffer = None
        
        # State
        self._is_initialized = False
    
    def _ensure_initialized(self) -> None:
        """Ensure engine is initialized."""
        if self._is_initialized:
            return
        
        # Open device if needed
        if not self.device._is_open:
            self.device.open()
        
        # Initialize tokenizer
        self._init_tokenizer()
        
        # Allocate device buffers
        self._allocate_buffers()
        
        self._is_initialized = True

    
    def _init_tokenizer(self) -> None:
        """Initialize the tokenizer."""
        if AutoTokenizer is None:
            logger.warning("transformers not installed, using fallback tokenizer")
            self._tokenizer = FallbackTokenizer()
            return
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name,
                trust_remote_code=True
            )
            logger.info(f"Loaded tokenizer: {self._tokenizer_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}, using fallback")
            self._tokenizer = FallbackTokenizer()
    
    def _allocate_buffers(self) -> None:
        """Allocate device memory buffers."""
        # Image buffer: 384x384x3 = 442,368 bytes
        img_size = IMG_SIZE * IMG_SIZE * 3
        self._image_buffer = self.device.alloc_dma_buffer(img_size)
        
        # Output buffer: max tokens * 4 bytes per token ID
        output_size = self.max_new_tokens * 4
        self._output_buffer = self.device.alloc_dma_buffer(output_size)
        
        # KV cache buffer: ~4MB for 2K context
        kv_cache_size = 4 * 1024 * 1024
        self._kv_cache_buffer = self.device.alloc_dma_buffer(kv_cache_size)
        
        # Configure hardware registers with buffer addresses
        self.device.write_reg(Registers.IMG_ADDR, self._image_buffer.physical_addr)
        self.device.write_reg(Registers.OUT_ADDR, self._output_buffer.physical_addr)
        self.device.write_reg(Registers.KV_CACHE_ADDR, self._kv_cache_buffer.physical_addr)
    
    def describe_image(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        prompt: str = "Describe this image in detail.",
        **kwargs
    ) -> InferenceResult:
        """
        Generate a description of an image.
        
        Args:
            image: Image path, numpy array, or PIL Image
            prompt: Text prompt (default: "Describe this image in detail.")
            **kwargs: Additional arguments passed to run()
            
        Returns:
            InferenceResult with generated description
        """
        return self.run(image, prompt, **kwargs)

    
    def run(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        callback: Optional[StreamingCallback] = None,
    ) -> InferenceResult:
        """
        Run vision-language inference.
        
        Args:
            image: Image input (path, numpy array, or PIL Image)
            prompt: Text prompt for the model
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            callback: Optional callback for streaming tokens
            
        Returns:
            InferenceResult containing output and timing info
        """
        self._ensure_initialized()
        timer = Timer()
        
        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature
        
        # Preprocess image
        timer.start("preprocess")
        img_data = self._preprocess_image(image)
        preprocess_time = timer.stop("preprocess")
        
        # Tokenize prompt
        timer.start("tokenize")
        input_ids = self._tokenize_prompt(prompt)
        tokenize_time = timer.stop("tokenize")
        
        # Upload image to device
        timer.start("upload")
        self._upload_image(img_data)
        timer.stop("upload")
        
        # Run inference
        timer.start("inference")
        tokens = self._run_hardware_inference(
            input_ids, 
            max_tokens, 
            temp,
            callback
        )
        inference_time = timer.stop("inference")
        
        # Decode output
        timer.start("decode")
        output_text = self._decode_tokens(tokens)
        timer.stop("decode")
        
        # Calculate timing
        total_time = timer.total_ms()
        vision_time = self._get_vision_time()
        generation_time = inference_time - vision_time
        tokens_per_sec = len(tokens) / (generation_time / 1000.0) if generation_time > 0 else 0
        
        return InferenceResult(
            text=output_text,
            tokens=tokens,
            num_tokens=len(tokens),
            vision_time_ms=vision_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            tokens_per_second=tokens_per_sec,
            preprocessing_time_ms=preprocess_time,
            tokenization_time_ms=tokenize_time,
            model_name=self.model_config.name if self.model_config else "unknown",
        )

    
    def stream(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Stream tokens as they are generated.
        
        Args:
            image: Image input
            prompt: Text prompt
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            
        Yields:
            Decoded text for each token
        """
        self._ensure_initialized()
        
        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature
        
        # Preprocess and upload image
        img_data = self._preprocess_image(image)
        self._upload_image(img_data)
        
        # Tokenize prompt
        input_ids = self._tokenize_prompt(prompt)
        
        # Start inference in streaming mode
        self._start_inference(input_ids, streaming=True)
        
        # Yield tokens as they become available
        tokens_generated = 0
        while tokens_generated < max_tokens:
            # Wait for token
            token = self._wait_for_token(timeout=5.0)
            if token is None:
                break
            
            # Check for EOS
            if token == self.EOS_TOKEN_ID:
                break
            
            # Decode and yield
            text = self._decode_tokens([token])
            yield text
            tokens_generated += 1
    
    def _preprocess_image(
        self, 
        image: Union[str, Path, np.ndarray, "Image.Image"]
    ) -> np.ndarray:
        """Preprocess image for hardware input."""
        # Load image if path
        if isinstance(image, (str, Path)):
            image = load_image(image)
        
        # Preprocess (resize, normalize)
        return preprocess_image(
            image, 
            size=IMG_SIZE,
            normalize=True,
            to_uint8=True
        )
    
    def _upload_image(self, img_data: np.ndarray) -> None:
        """Upload preprocessed image to device."""
        # Ensure correct shape and dtype
        if img_data.shape != (IMG_SIZE, IMG_SIZE, 3):
            raise ValueError(f"Image must be {IMG_SIZE}x{IMG_SIZE}x3, got {img_data.shape}")
        
        flat_data = img_data.flatten().astype(np.uint8)
        self._image_buffer.write(flat_data)
        
        # DMA transfer to device
        self.device.dma_transfer_to_device(
            self._image_buffer,
            self.device.read_reg(Registers.IMG_ADDR),
            flat_data.nbytes
        )
        
        # Set image size register
        self.device.write_reg(Registers.IMG_SIZE, (IMG_SIZE << 16) | IMG_SIZE)

    
    def _tokenize_prompt(self, prompt: str) -> List[int]:
        """Tokenize text prompt."""
        # Format prompt with image token
        formatted_prompt = f"{self.IMAGE_TOKEN}\n{prompt}"
        
        if hasattr(self._tokenizer, 'encode'):
            return self._tokenizer.encode(formatted_prompt, add_special_tokens=True)
        else:
            return self._tokenizer.tokenize(formatted_prompt)
    
    def _decode_tokens(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        if hasattr(self._tokenizer, 'decode'):
            return self._tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            return self._tokenizer.detokenize(tokens)
    
    def _start_inference(self, input_ids: List[int], streaming: bool = False) -> None:
        """Start hardware inference."""
        # Upload input tokens to device memory
        # For now, tokens are part of the command - in real impl would use DMA
        
        ctrl = CtrlBits.ENABLE | CtrlBits.START_INFERENCE
        if streaming:
            ctrl |= CtrlBits.STREAMING_MODE
        
        self.device.write_reg(Registers.CTRL, ctrl)
    
    def _run_hardware_inference(
        self,
        input_ids: List[int],
        max_tokens: int,
        temperature: float,
        callback: Optional[StreamingCallback] = None,
    ) -> List[int]:
        """Run inference and collect output tokens."""
        self._start_inference(input_ids, streaming=callback is not None)
        
        tokens = []
        
        # For simulated device, handle specially
        if isinstance(self.device, SimulatedDevice):
            return self._run_simulated_inference(max_tokens, temperature, callback)
        
        # Wait for hardware completion
        timeout = 30.0  # 30 second timeout
        start = time.time()
        
        while len(tokens) < max_tokens and (time.time() - start) < timeout:
            status = self.device.get_status()
            
            if status & StatusBits.ERROR:
                raise DeviceError("Hardware inference error")
            
            if status & StatusBits.TOKEN_READY:
                token = self.device.read_reg(Registers.TOKEN_OUT)
                tokens.append(token)
                
                if callback:
                    text = self._decode_tokens([token])
                    callback(text)
                
                if token == self.EOS_TOKEN_ID:
                    break
            
            if status & StatusBits.LLM_DONE:
                break
            
            time.sleep(0.001)
        
        return tokens

    
    def _run_simulated_inference(
        self,
        max_tokens: int,
        temperature: float,
        callback: Optional[StreamingCallback] = None,
    ) -> List[int]:
        """Run simulated inference (for development without hardware)."""
        tokens = []
        sim_device = self.device
        
        while len(tokens) < max_tokens:
            # Wait for simulated token
            time.sleep(0.001)
            
            token = sim_device.get_next_token()
            if token is None:
                status = sim_device.get_status()
                if status & StatusBits.LLM_DONE:
                    break
                continue
            
            tokens.append(token)
            
            if callback:
                text = self._decode_tokens([token])
                callback(text)
            
            if token == self.EOS_TOKEN_ID:
                break
        
        return tokens
    
    def _wait_for_token(self, timeout: float = 5.0) -> Optional[int]:
        """Wait for next token to be ready."""
        start = time.time()
        
        while time.time() - start < timeout:
            status = self.device.get_status()
            
            if status & StatusBits.ERROR:
                raise DeviceError("Hardware error during inference")
            
            if status & StatusBits.TOKEN_READY:
                return self.device.read_reg(Registers.TOKEN_OUT)
            
            if status & StatusBits.LLM_DONE:
                return None
            
            time.sleep(0.001)
        
        raise DeviceTimeoutError("Timeout waiting for token")
    
    def _get_vision_time(self) -> float:
        """Get vision processing time from hardware."""
        # Would read from debug registers in real implementation
        return 5.0  # Placeholder
    
    def close(self) -> None:
        """Clean up resources."""
        if self._image_buffer:
            self.device.free_dma_buffer(self._image_buffer)
        if self._output_buffer:
            self.device.free_dma_buffer(self._output_buffer)
        if self._kv_cache_buffer:
            self.device.free_dma_buffer(self._kv_cache_buffer)
        self._is_initialized = False

    # =========================================================================
    # Async Interface
    # =========================================================================

    async def run_async(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> InferenceResult:
        """
        Run vision-language inference asynchronously.
        
        Args:
            image: Image input (path, numpy array, or PIL Image)
            prompt: Text prompt for the model
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            
        Returns:
            InferenceResult containing output and timing info
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.run(image, prompt, max_new_tokens, temperature)
        )
    
    async def describe_image_async(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        prompt: str = "Describe this image in detail.",
        **kwargs
    ) -> InferenceResult:
        """
        Generate a description of an image asynchronously.
        
        Args:
            image: Image path, numpy array, or PIL Image
            prompt: Text prompt (default: "Describe this image in detail.")
            **kwargs: Additional arguments passed to run_async()
            
        Returns:
            InferenceResult with generated description
        """
        return await self.run_async(image, prompt, **kwargs)
    
    async def stream_async(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens asynchronously as they are generated.
        
        Args:
            image: Image input
            prompt: Text prompt
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            
        Yields:
            Decoded text for each token
        """
        self._ensure_initialized()
        
        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature
        
        # Preprocess and upload image (run in executor)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._preprocess_and_upload(image))
        
        # Tokenize prompt
        input_ids = await loop.run_in_executor(
            None, lambda: self._tokenize_prompt(prompt)
        )
        
        # Start inference in streaming mode
        await loop.run_in_executor(
            None, lambda: self._start_inference(input_ids, streaming=True)
        )
        
        # Yield tokens as they become available
        tokens_generated = 0
        while tokens_generated < max_tokens:
            # Wait for token (with async sleep to not block)
            token = await loop.run_in_executor(
                None, lambda: self._wait_for_token(timeout=0.1)
            )
            
            if token is None:
                # Check if done
                status = self.device.get_status()
                if status & StatusBits.LLM_DONE:
                    break
                # Small async delay before retry
                await asyncio.sleep(0.001)
                continue
            
            # Check for EOS
            if token == self.EOS_TOKEN_ID:
                break
            
            # Decode and yield
            text = self._decode_tokens([token])
            yield text
            tokens_generated += 1
    
    def _preprocess_and_upload(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"]
    ) -> None:
        """Preprocess image and upload to device (helper for async)."""
        img_data = self._preprocess_image(image)
        self._upload_image(img_data)


class FallbackTokenizer:
    """Simple fallback tokenizer when transformers is not available."""
    
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
    
    def tokenize(self, text: str) -> List[int]:
        """Simple character-level tokenization."""
        return [ord(c) % self.vocab_size for c in text]
    
    def detokenize(self, tokens: List[int]) -> str:
        """Decode tokens back to text."""
        return "".join(chr(t) if 32 <= t < 127 else "?" for t in tokens)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [1] + tokens  # BOS token
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in (0, 1, 2)]
        return self.detokenize(tokens)
