"""
SiLens Model Backend - Real model inference for simulated device.

This module provides a backend that runs actual SmolVLM-256M inference
using HuggingFace transformers, simulating what the hardware would do.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Iterator, Union, Callable
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Try to import required libraries
_HAS_TRANSFORMERS = False
_HAS_TORCH = False
_HAS_PIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    _HAS_TRANSFORMERS = True
except ImportError:
    AutoProcessor = None
    AutoModelForVision2Seq = None

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    Image = None


@dataclass
class ModelBackendConfig:
    """Configuration for the model backend."""
    model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


class SmolVLMBackend:
    """
    Backend that runs actual SmolVLM-256M model inference.
    
    This provides real model outputs for the simulated device,
    allowing end-to-end testing without hardware.
    """
    
    def __init__(self, config: Optional[ModelBackendConfig] = None):
        self.config = config or ModelBackendConfig()
        self._model = None
        self._processor = None
        self._device = None
        self._is_loaded = False
        self._lock = threading.Lock()
        
    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        missing = []
        if not _HAS_TORCH:
            missing.append("torch")
        if not _HAS_TRANSFORMERS:
            missing.append("transformers")
        if not _HAS_PIL:
            missing.append("pillow")
        
        if missing:
            raise ImportError(
                f"Missing required packages for model backend: {', '.join(missing)}\n"
                f"Install with: pip install {' '.join(missing)}"
            )
    
    def load(self) -> None:
        """Load the model and processor."""
        if self._is_loaded:
            return
            
        self._check_dependencies()
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Determine device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = self.config.device
        
        logger.info(f"Using device: {self._device}")
        
        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Load model
        dtype = torch.float16 if self._device in ("cuda", "mps") else torch.float32
        
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self._device)
        
        self._model.eval()
        self._is_loaded = True
        logger.info("Model loaded successfully")
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        logger.info("Model unloaded")
    
    def generate(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        prompt: str,
        max_new_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """
        Generate text response for an image and prompt.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            prompt: Text prompt
            max_new_tokens: Override default max tokens
            stream: If True, yield tokens as they're generated
            
        Returns:
            Generated text string, or iterator of token strings if stream=True
        """
        if not self._is_loaded:
            self.load()
        
        max_tokens = max_new_tokens or self.config.max_new_tokens
        
        # Load image
        pil_image = self._load_image(image)
        
        # Format messages for SmolVLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text_prompt = self._processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self._processor(
            text=text_prompt,
            images=[pil_image],
            return_tensors="pt"
        ).to(self._device)
        
        if stream:
            return self._stream_generate(inputs, max_tokens)
        else:
            return self._batch_generate(inputs, max_tokens)
    
    def _load_image(
        self, 
        image: Union[str, Path, np.ndarray, "Image.Image"]
    ) -> "Image.Image":
        """Load and convert image to PIL format."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif hasattr(image, "convert"):  # PIL Image
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _batch_generate(self, inputs, max_tokens: int) -> str:
        """Generate complete response at once."""
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        
        # Decode only the generated tokens (exclude input)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        
        return self._processor.decode(generated_ids, skip_special_tokens=True)
    
    def _stream_generate(self, inputs, max_tokens: int) -> Iterator[str]:
        """Generate tokens one at a time for streaming."""
        from transformers import TextIteratorStreamer
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generation kwargs
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "streamer": streamer,
        }
        
        # Start generation in background thread
        thread = threading.Thread(
            target=self._model.generate,
            kwargs=gen_kwargs
        )
        thread.start()
        
        # Yield tokens as they come
        for text in streamer:
            yield text
        
        thread.join()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @property
    def device_name(self) -> str:
        """Get the device being used."""
        return self._device or "not loaded"


class ModelBackedSimulatedDevice:
    """
    Simulated SiLens device that uses real SmolVLM model for inference.
    
    This combines the simulated device interface with actual model inference,
    providing a realistic end-to-end experience without hardware.
    
    Example:
        device = ModelBackedSimulatedDevice()
        device.open()  # Loads the model
        
        result = device.generate(image, prompt)
        print(result)
        
        # Streaming
        for token in device.stream(image, prompt):
            print(token, end="", flush=True)
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        device: str = "auto",
    ):
        self.backend_config = ModelBackendConfig(
            model_name=model_name,
            device=device,
        )
        self._backend: Optional[SmolVLMBackend] = None
        self._is_open = False
        
        # Timing simulation
        self._last_vision_time_ms = 0.0
        self._last_generation_time_ms = 0.0
    
    def open(self) -> None:
        """Open device and load model."""
        if self._is_open:
            return
        
        self._backend = SmolVLMBackend(self.backend_config)
        self._backend.load()
        self._is_open = True
    
    def close(self) -> None:
        """Close device and unload model."""
        if self._backend:
            self._backend.unload()
            self._backend = None
        self._is_open = False
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def generate(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate text response for image and prompt.
        
        Args:
            image: Input image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text string
        """
        if not self._is_open:
            raise RuntimeError("Device not open. Call open() first.")
        
        start = time.perf_counter()
        result = self._backend.generate(
            image, prompt, 
            max_new_tokens=max_new_tokens,
            stream=False
        )
        elapsed = (time.perf_counter() - start) * 1000
        
        # Estimate timing breakdown
        self._last_vision_time_ms = elapsed * 0.2  # ~20% for vision
        self._last_generation_time_ms = elapsed * 0.8  # ~80% for LLM
        
        return result
    
    def stream(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        prompt: str,
        max_new_tokens: int = 256,
    ) -> Iterator[str]:
        """
        Stream generated tokens.
        
        Args:
            image: Input image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            
        Yields:
            Token strings as they're generated
        """
        if not self._is_open:
            raise RuntimeError("Device not open. Call open() first.")
        
        start = time.perf_counter()
        first_token = True
        
        for token in self._backend.generate(
            image, prompt,
            max_new_tokens=max_new_tokens,
            stream=True
        ):
            if first_token:
                self._last_vision_time_ms = (time.perf_counter() - start) * 1000
                first_token = False
            yield token
        
        self._last_generation_time_ms = (time.perf_counter() - start) * 1000 - self._last_vision_time_ms
    
    def get_timing(self) -> dict:
        """Get timing from last inference."""
        return {
            "vision_time_ms": self._last_vision_time_ms,
            "generation_time_ms": self._last_generation_time_ms,
            "total_time_ms": self._last_vision_time_ms + self._last_generation_time_ms,
        }
    
    @property
    def is_open(self) -> bool:
        return self._is_open
    
    @property
    def model_name(self) -> str:
        return self.backend_config.model_name
    
    @property 
    def device_name(self) -> str:
        if self._backend:
            return self._backend.device_name
        return "not loaded"


def check_backend_available() -> bool:
    """Check if the model backend dependencies are available."""
    return _HAS_TORCH and _HAS_TRANSFORMERS and _HAS_PIL


def get_backend_status() -> dict:
    """Get status of backend dependencies."""
    return {
        "torch": _HAS_TORCH,
        "transformers": _HAS_TRANSFORMERS,
        "pillow": _HAS_PIL,
        "available": check_backend_available(),
    }
