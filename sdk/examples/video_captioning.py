#!/usr/bin/env python3
"""
SiLens Video Captioning Example.

Demonstrates how to caption video frames using the SiLens SDK.
Extracts frames at configurable intervals and generates captions.

Usage:
    python video_captioning.py video.mp4 --output captions.json
    python video_captioning.py video.mp4 --fps 1 --prompt "What is happening?"
    python video_captioning.py video.mp4 --simulation --verbose

Requirements:
    pip install opencv-python
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple
import numpy as np

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not installed. Install with: pip install opencv-python")

from silens import SiLensDevice, InferenceEngine


def extract_frames(
    video_path: str,
    fps: float = 1.0,
    max_frames: int = None,
) -> Iterator[Tuple[float, np.ndarray]]:
    """
    Extract frames from a video at the specified FPS.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to extract
        max_frames: Maximum frames to extract
        
    Yields:
        Tuple of (timestamp_sec, frame_array)
    """
    if not HAS_OPENCV:
        raise ImportError("OpenCV is required: pip install opencv-python")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"Video: {video_path}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  FPS: {video_fps:.1f}")
    print(f"  Total frames: {total_frames}")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps) if fps < video_fps else 1
    
    frame_count = 0
    extracted = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield (timestamp, frame_rgb)
            extracted += 1
            
            if max_frames and extracted >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"  Extracted: {extracted} frames")


def caption_video(
    engine: InferenceEngine,
    video_path: str,
    prompt: str,
    fps: float = 1.0,
    max_frames: int = None,
    max_tokens: int = 64,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate captions for video frames.
    
    Args:
        engine: InferenceEngine instance
        video_path: Path to video file
        prompt: Prompt for each frame
        fps: Frames per second to caption
        max_frames: Maximum frames to process
        max_tokens: Max tokens per caption
        verbose: Show verbose output
        
    Returns:
        List of caption dictionaries
    """
    captions = []
    
    print(f"\nProcessing video at {fps} FPS...")
    print(f"Prompt: {prompt}")
    print()
    
    start_time = time.time()
    
    for timestamp, frame in extract_frames(video_path, fps, max_frames):
        try:
            result = engine.run(frame, prompt, max_new_tokens=max_tokens)
            
            caption = {
                "timestamp_sec": round(timestamp, 2),
                "timestamp_str": format_timestamp(timestamp),
                "caption": result.text.strip(),
                "num_tokens": result.num_tokens,
                "inference_ms": result.total_time_ms,
            }
            captions.append(caption)
            
            if verbose:
                print(f"[{caption['timestamp_str']}] {caption['caption'][:60]}...")
            else:
                print(f".", end="", flush=True)
                
        except Exception as e:
            captions.append({
                "timestamp_sec": round(timestamp, 2),
                "timestamp_str": format_timestamp(timestamp),
                "error": str(e),
            })
            if verbose:
                print(f"[{format_timestamp(timestamp)}] ERROR: {e}")
    
    if not verbose:
        print()  # Newline after dots
    
    elapsed = time.time() - start_time
    print(f"\nProcessed {len(captions)} frames in {elapsed:.1f}s")
    
    return captions


def format_timestamp(seconds: float) -> str:
    """Format timestamp as MM:SS.mmm"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def generate_srt(captions: List[Dict], duration_per_caption: float = 3.0) -> str:
    """Generate SRT subtitle format."""
    lines = []
    
    for i, cap in enumerate(captions, 1):
        if "error" in cap:
            continue
        
        start = cap["timestamp_sec"]
        end = start + duration_per_caption
        
        # Next caption's start time limits this caption's end time
        if i < len(captions):
            next_start = captions[i]["timestamp_sec"] if i < len(captions) else end
            end = min(end, next_start)
        
        start_str = format_srt_time(start)
        end_str = format_srt_time(end)
        
        lines.append(str(i))
        lines.append(f"{start_str} --> {end_str}")
        lines.append(cap["caption"])
        lines.append("")
    
    return "\n".join(lines)


def format_srt_time(seconds: float) -> str:
    """Format time for SRT (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Caption video frames using SiLens"
    )
    parser.add_argument(
        "video",
        type=str,
        help="Input video file",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Describe what is happening in this frame.",
        help="Prompt for each frame",
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=1.0,
        help="Frames per second to caption (default: 1.0)",
    )
    parser.add_argument(
        "--max-frames", "-n",
        type=int,
        help="Maximum frames to process",
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=64,
        help="Maximum tokens per caption (default: 64)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file (JSON or SRT based on extension)",
    )
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Force simulation mode",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output",
    )
    
    args = parser.parse_args()
    
    if not HAS_OPENCV:
        print("Error: OpenCV is required. Install with: pip install opencv-python")
        sys.exit(1)
    
    # Check video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    # Discover device
    mode = "simulation" if args.simulation else "auto"
    devices = SiLensDevice.discover(mode=mode)
    device = devices[0]
    print(f"Using device: {device}\n")
    
    # Process video
    with device:
        engine = InferenceEngine(device, max_new_tokens=args.max_tokens)
        
        captions = caption_video(
            engine,
            str(video_path),
            args.prompt,
            fps=args.fps,
            max_frames=args.max_frames,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )
        
        engine.close()
    
    # Save output
    if args.output:
        output_path = Path(args.output)
        
        if output_path.suffix.lower() == ".srt":
            output_path.write_text(generate_srt(captions))
            print(f"Subtitles saved to: {output_path}")
        else:
            output_data = {
                "video": str(video_path),
                "prompt": args.prompt,
                "fps": args.fps,
                "captions": captions,
            }
            output_path.write_text(json.dumps(output_data, indent=2))
            print(f"Captions saved to: {output_path}")
    else:
        # Print captions to stdout
        print("\n" + "=" * 60)
        print("Video Captions")
        print("=" * 60)
        for cap in captions:
            if "error" in cap:
                print(f"[{cap['timestamp_str']}] ERROR: {cap['error']}")
            else:
                print(f"[{cap['timestamp_str']}] {cap['caption']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
