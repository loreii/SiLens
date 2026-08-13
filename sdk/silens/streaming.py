"""
SiLens Streaming Module.

Streaming inference support for token-by-token generation with:
- Async streaming interface
- Callback-based streaming
- Server-Sent Events (SSE) compatible output
- Buffered text decoding for proper word boundaries
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Optional, Callable, Iterator, AsyncIterator, 
    List, Dict, Any, Union, TYPE_CHECKING
)

if TYPE_CHECKING:
    from silens.device import SiLensDevice
    from silens.inference import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming inference."""
    buffer_size: int = 10
    timeout_ms: float = 5000.0
    min_tokens_before_yield: int = 1
    flush_on_newline: bool = True
    flush_on_punctuation: bool = True
    word_boundary_tokens: bool = True


@dataclass
class StreamingToken:
    """Represents a single streamed token."""
    token_id: int
    text: str
    logprob: Optional[float] = None
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)
    is_special: bool = False
    is_eos: bool = False


@dataclass
class StreamingStats:
    """Statistics for a streaming session."""
    total_tokens: int = 0
    start_time_ms: float = 0.0
    end_time_ms: float = 0.0
    first_token_time_ms: float = 0.0
    
    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        return self.end_time_ms - self.start_time_ms
    
    @property
    def time_to_first_token_ms(self) -> float:
        """Time to first token in milliseconds."""
        return self.first_token_time_ms - self.start_time_ms
    
    @property
    def tokens_per_second(self) -> float:
        """Token generation rate."""
        if self.duration_ms <= 0:
            return 0.0
        return self.total_tokens / (self.duration_ms / 1000.0)


class StreamingCallback(ABC):
    """Abstract base class for streaming callbacks."""
    
    @abstractmethod
    def on_token(self, token: StreamingToken) -> bool:
        """
        Called for each generated token.
        
        Args:
            token: The generated token
            
        Returns:
            True to continue generation, False to stop
        """
        pass
    
    def on_start(self) -> None:
        """Called when streaming starts."""
        pass
    
    def on_end(self, stats: StreamingStats) -> None:
        """Called when streaming ends."""
        pass
    
    def on_error(self, error: Exception) -> None:
        """Called on error."""
        pass


class PrintCallback(StreamingCallback):
    """Simple callback that prints tokens to stdout."""
    
    def __init__(self, end: str = "", flush: bool = True):
        self.end = end
        self.flush = flush
    
    def on_token(self, token: StreamingToken) -> bool:
        if not token.is_special:
            print(token.text, end=self.end, flush=self.flush)
        return True
    
    def on_end(self, stats: StreamingStats) -> None:
        print()  # Newline at end


class BufferCallback(StreamingCallback):
    """Callback that buffers all tokens."""
    
    def __init__(self):
        self.tokens: List[StreamingToken] = []
        self.text_buffer: List[str] = []
    
    def on_token(self, token: StreamingToken) -> bool:
        self.tokens.append(token)
        if not token.is_special:
            self.text_buffer.append(token.text)
        return True
    
    @property
    def text(self) -> str:
        """Get complete generated text."""
        return "".join(self.text_buffer)


class TokenBuffer:
    """
    Buffer for accumulating tokens and yielding complete text chunks.
    
    Helps produce smoother output by yielding text at word boundaries
    rather than after each token.
    """
    
    PUNCTUATION = {'.', '!', '?', ',', ';', ':', '\n'}
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self._buffer: List[str] = []
        self._pending = ""
    
    def add(self, text: str) -> Optional[str]:
        """
        Add text to buffer.
        
        Args:
            text: Token text to add
            
        Returns:
            Text to yield, or None if still buffering
        """
        self._pending += text
        
        # Check flush conditions
        should_flush = False
        
        if self.config.flush_on_newline and '\n' in self._pending:
            should_flush = True
        elif self.config.flush_on_punctuation:
            if any(c in self._pending for c in self.PUNCTUATION):
                should_flush = True
        elif self.config.word_boundary_tokens and ' ' in self._pending:
            should_flush = True
        
        if should_flush:
            result = self._pending
            self._pending = ""
            return result
        
        return None
    
    def flush(self) -> str:
        """Flush remaining buffer contents."""
        result = self._pending
        self._pending = ""
        return result


class StreamingGenerator:
    """
    Generator for streaming token output from SiLens device.
    
    Wraps the hardware token generation and provides clean iteration
    interfaces for both sync and async use.
    
    Example:
        generator = StreamingGenerator(device)
        
        # Synchronous iteration
        for token in generator.generate(input_ids):
            print(token.text, end="", flush=True)
        
        # Async iteration
        async for token in generator.generate_async(input_ids):
            yield token.text
    """
    
    EOS_TOKEN_ID = 2
    
    def __init__(
        self,
        device: "SiLensDevice",
        tokenizer: Any = None,
        config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize streaming generator.
        
        Args:
            device: SiLens device instance
            tokenizer: Tokenizer for decoding tokens
            config: Streaming configuration
        """
        self.device = device
        self.tokenizer = tokenizer
        self.config = config or StreamingConfig()
        self._stop_event = threading.Event()
    
    def stop(self) -> None:
        """Signal to stop generation."""
        self._stop_event.set()

    
    def generate(
        self,
        input_ids: List[int],
        max_tokens: int = 256,
        callback: Optional[StreamingCallback] = None,
    ) -> Iterator[StreamingToken]:
        """
        Generate tokens synchronously.
        
        Args:
            input_ids: Input token IDs
            max_tokens: Maximum tokens to generate
            callback: Optional callback for each token
            
        Yields:
            StreamingToken for each generated token
        """
        self._stop_event.clear()
        stats = StreamingStats(start_time_ms=time.time() * 1000)
        
        if callback:
            callback.on_start()
        
        try:
            # Start inference on device
            self._start_device_inference(input_ids)
            
            tokens_generated = 0
            first_token = True
            
            while tokens_generated < max_tokens and not self._stop_event.is_set():
                # Wait for token from device
                token_id = self._wait_for_token()
                
                if token_id is None:
                    break
                
                # Record first token time
                if first_token:
                    stats.first_token_time_ms = time.time() * 1000
                    first_token = False
                
                # Decode token
                text = self._decode_token(token_id)
                is_eos = token_id == self.EOS_TOKEN_ID
                
                token = StreamingToken(
                    token_id=token_id,
                    text=text,
                    is_eos=is_eos,
                    is_special=token_id < 10,
                )
                
                # Call callback
                if callback:
                    if not callback.on_token(token):
                        break
                
                yield token
                tokens_generated += 1
                
                if is_eos:
                    break
            
            stats.total_tokens = tokens_generated
            stats.end_time_ms = time.time() * 1000
            
            if callback:
                callback.on_end(stats)
                
        except Exception as e:
            if callback:
                callback.on_error(e)
            raise
    
    async def generate_async(
        self,
        input_ids: List[int],
        max_tokens: int = 256,
        callback: Optional[StreamingCallback] = None,
    ) -> AsyncIterator[StreamingToken]:
        """
        Generate tokens asynchronously.
        
        Args:
            input_ids: Input token IDs
            max_tokens: Maximum tokens to generate
            callback: Optional callback for each token
            
        Yields:
            StreamingToken for each generated token
        """
        self._stop_event.clear()
        stats = StreamingStats(start_time_ms=time.time() * 1000)
        
        if callback:
            callback.on_start()
        
        try:
            # Start inference on device
            await asyncio.get_event_loop().run_in_executor(
                None, self._start_device_inference, input_ids
            )
            
            tokens_generated = 0
            first_token = True
            
            while tokens_generated < max_tokens and not self._stop_event.is_set():
                # Wait for token from device (async)
                token_id = await asyncio.get_event_loop().run_in_executor(
                    None, self._wait_for_token
                )
                
                if token_id is None:
                    break
                
                if first_token:
                    stats.first_token_time_ms = time.time() * 1000
                    first_token = False
                
                text = self._decode_token(token_id)
                is_eos = token_id == self.EOS_TOKEN_ID
                
                token = StreamingToken(
                    token_id=token_id,
                    text=text,
                    is_eos=is_eos,
                    is_special=token_id < 10,
                )
                
                if callback:
                    if not callback.on_token(token):
                        break
                
                yield token
                tokens_generated += 1
                
                if is_eos:
                    break
            
            stats.total_tokens = tokens_generated
            stats.end_time_ms = time.time() * 1000
            
            if callback:
                callback.on_end(stats)
                
        except Exception as e:
            if callback:
                callback.on_error(e)
            raise

    
    def _start_device_inference(self, input_ids: List[int]) -> None:
        """Start inference on the device."""
        from silens.device import CtrlBits, Registers
        
        ctrl = CtrlBits.ENABLE | CtrlBits.START_INFERENCE | CtrlBits.STREAMING_MODE
        self.device.write_reg(Registers.CTRL, ctrl)
    
    def _wait_for_token(self) -> Optional[int]:
        """Wait for next token from device."""
        from silens.device import StatusBits, Registers
        
        timeout_sec = self.config.timeout_ms / 1000.0
        start = time.time()
        
        while time.time() - start < timeout_sec:
            if self._stop_event.is_set():
                return None
            
            status = self.device.get_status()
            
            if status & StatusBits.ERROR:
                raise RuntimeError("Device error during inference")
            
            if status & StatusBits.TOKEN_READY:
                return self.device.read_reg(Registers.TOKEN_OUT)
            
            if status & StatusBits.LLM_DONE:
                return None
            
            time.sleep(0.001)
        
        return None
    
    def _decode_token(self, token_id: int) -> str:
        """Decode a single token ID."""
        if self.tokenizer is None:
            # Fallback: simple character mapping
            if 32 <= token_id < 127:
                return chr(token_id)
            return ""
        
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode([token_id], skip_special_tokens=True)
        return ""


class SSEFormatter:
    """
    Server-Sent Events (SSE) formatter for streaming.
    
    Formats streaming tokens as SSE events for HTTP streaming responses.
    
    Example:
        formatter = SSEFormatter()
        for token in generator.generate(input_ids):
            yield formatter.format(token)
        yield formatter.done()
    """
    
    def format(self, token: StreamingToken) -> str:
        """Format a token as an SSE event."""
        import json
        data = {
            "token": token.text,
            "token_id": token.token_id,
            "timestamp": token.timestamp_ms,
        }
        if token.logprob is not None:
            data["logprob"] = token.logprob
        return f"data: {json.dumps(data)}\n\n"
    
    def done(self) -> str:
        """Return the done event."""
        return "data: [DONE]\n\n"
    
    def error(self, message: str) -> str:
        """Format an error event."""
        import json
        return f"data: {json.dumps({'error': message})}\n\n"


# Async wrapper for existing InferenceEngine

class AsyncInferenceWrapper:
    """
    Async wrapper for InferenceEngine.
    
    Provides async/await interface for inference operations.
    
    Example:
        wrapper = AsyncInferenceWrapper(engine)
        
        result = await wrapper.run(image, prompt)
        
        async for token in wrapper.stream(image, prompt):
            print(token, end="")
    """
    
    def __init__(self, engine: "InferenceEngine"):
        self.engine = engine
        self._executor = None
    
    async def run(self, image, prompt: str, **kwargs):
        """Run inference asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.engine.run(image, prompt, **kwargs)
        )
    
    async def describe_image(self, image, prompt: str = "Describe this image.", **kwargs):
        """Describe an image asynchronously."""
        return await self.run(image, prompt, **kwargs)
    
    async def stream(
        self,
        image,
        prompt: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream inference results asynchronously.
        
        Args:
            image: Input image
            prompt: Text prompt
            **kwargs: Additional arguments
            
        Yields:
            Token text strings
        """
        # Use a queue to bridge sync generator to async
        token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        
        def producer():
            try:
                for token_text in self.engine.stream(image, prompt, **kwargs):
                    asyncio.run_coroutine_threadsafe(
                        token_queue.put(token_text),
                        asyncio.get_event_loop()
                    )
            finally:
                asyncio.run_coroutine_threadsafe(
                    token_queue.put(None),
                    asyncio.get_event_loop()
                )
        
        # Start producer in thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, producer)
        
        # Consume from queue
        while True:
            token = await token_queue.get()
            if token is None:
                break
            yield token
