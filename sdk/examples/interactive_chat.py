#!/usr/bin/env python3
"""
SiLens Interactive Chat Example.

Provides an interactive command-line chat session with the SiLens VLM.
Supports loading images, asking questions, and streaming responses.

Usage:
    python interactive_chat.py
    python interactive_chat.py --image photo.jpg
    python interactive_chat.py --simulation

Commands in chat:
    /image PATH    - Load a new image
    /clear         - Clear conversation history
    /stats         - Show session statistics
    /stream        - Toggle streaming mode
    /help          - Show help
    /quit          - Exit the chat
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from silens import SiLensDevice, InferenceEngine, load_image


class ChatSession:
    """Interactive chat session with SiLens VLM."""
    
    def __init__(
        self,
        engine: InferenceEngine,
        initial_image: Optional[str] = None,
        streaming: bool = True,
    ):
        self.engine = engine
        self.current_image = None
        self.streaming = streaming
        self.history: List[Dict[str, Any]] = []
        self.stats = {
            "messages": 0,
            "total_tokens": 0,
            "total_time_ms": 0,
        }
        
        if initial_image:
            self.load_image(initial_image)
    
    def load_image(self, path: str) -> bool:
        """Load an image for the conversation."""
        try:
            image_path = Path(path)
            if not image_path.exists():
                print(f"Error: Image not found: {path}")
                return False
            
            self.current_image = str(image_path)
            print(f"✅ Loaded image: {image_path.name}")
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        print("✅ Conversation history cleared")

    
    def show_stats(self) -> None:
        """Show session statistics."""
        print("\n📊 Session Statistics:")
        print(f"  Messages: {self.stats['messages']}")
        print(f"  Total tokens: {self.stats['total_tokens']}")
        print(f"  Total time: {self.stats['total_time_ms']:.1f} ms")
        if self.stats['messages'] > 0:
            avg = self.stats['total_time_ms'] / self.stats['messages']
            print(f"  Avg response time: {avg:.1f} ms")
        if self.current_image:
            print(f"  Current image: {self.current_image}")
        print()
    
    def toggle_streaming(self) -> None:
        """Toggle streaming mode."""
        self.streaming = not self.streaming
        mode = "ON" if self.streaming else "OFF"
        print(f"✅ Streaming mode: {mode}")
    
    def chat(self, message: str) -> Optional[str]:
        """Send a message and get a response."""
        if not self.current_image:
            print("⚠️  No image loaded. Use /image PATH to load an image.")
            return None
        
        start = time.time()
        
        if self.streaming:
            # Stream response
            print("\n🤖 Assistant: ", end="", flush=True)
            
            tokens = []
            for token_text in self.engine.stream(
                self.current_image,
                message,
            ):
                print(token_text, end="", flush=True)
                tokens.append(token_text)
            
            print()
            response = "".join(tokens)
            
        else:
            # Single response
            result = self.engine.run(self.current_image, message)
            response = result.text
            print(f"\n🤖 Assistant: {response}")
        
        elapsed = (time.time() - start) * 1000
        num_tokens = len(response.split())  # Approximate
        
        # Update history and stats
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        
        self.stats["messages"] += 1
        self.stats["total_tokens"] += num_tokens
        self.stats["total_time_ms"] += elapsed
        
        return response
    
    def handle_command(self, command: str) -> bool:
        """
        Handle a chat command.
        
        Returns True if the chat should continue, False to exit.
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd in ("/quit", "/exit", "/q"):
            print("Goodbye! 👋")
            return False
        
        elif cmd in ("/image", "/img", "/load"):
            if not arg:
                print("Usage: /image PATH")
            else:
                self.load_image(arg)
        
        elif cmd in ("/clear", "/reset"):
            self.clear_history()
        
        elif cmd in ("/stats", "/info"):
            self.show_stats()
        
        elif cmd in ("/stream", "/streaming"):
            self.toggle_streaming()
        
        elif cmd in ("/help", "/?"):
            self.show_help()
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands")
        
        return True
    
    def show_help(self) -> None:
        """Show help message."""
        print("""
📖 Available Commands:
  /image PATH  - Load a new image
  /clear       - Clear conversation history
  /stats       - Show session statistics
  /stream      - Toggle streaming mode
  /help        - Show this help
  /quit        - Exit the chat

💡 Tips:
  - Load an image first with /image
  - Ask any question about the image
  - Use streaming mode for real-time output
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with SiLens VLM"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Initial image to load",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming mode",
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=256,
        help="Maximum tokens per response (default: 256)",
    )
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Force simulation mode",
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("""
╔═══════════════════════════════════════════════════════════╗
║               SiLens Interactive Chat                     ║
║         Vision-Language AI at your fingertips             ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    # Discover device
    print("🔌 Connecting to device...")
    mode = "simulation" if args.simulation else "auto"
    
    try:
        devices = SiLensDevice.discover(mode=mode)
        device = devices[0]
        print(f"   Using: {device}\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Start chat
    with device:
        engine = InferenceEngine(device, max_new_tokens=args.max_tokens)
        
        session = ChatSession(
            engine,
            initial_image=args.image,
            streaming=not args.no_stream,
        )
        
        if not args.image:
            print("💡 Tip: Load an image with /image PATH or start a conversation\n")
        
        session.show_help()
        
        # Main loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    if not session.handle_command(user_input):
                        break
                else:
                    session.chat(user_input)
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except EOFError:
                print("\nGoodbye! 👋")
                break
        
        engine.close()


if __name__ == "__main__":
    main()
