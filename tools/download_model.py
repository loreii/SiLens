#!/usr/bin/env python3
"""
Download SmolVLM-256M model from Hugging Face Hub.

This script downloads the model and processor to the local model/ directory.
The model is approximately 500MB in size.

Usage:
    python tools/download_model.py
    python tools/download_model.py --output-dir ./my_models
"""

import argparse
import os
import sys
from pathlib import Path


def download_model(output_dir: str = "model/smolvlm-256m", force: bool = False):
    """
    Download SmolVLM-256M-Instruct model from Hugging Face.
    
    Args:
        output_dir: Directory to save the model
        force: If True, re-download even if model exists
    """
    output_path = Path(output_dir)
    
    # Check if already downloaded
    if output_path.exists() and not force:
        config_file = output_path / "config.json"
        if config_file.exists():
            print(f"✓ Model already exists at {output_path}")
            print("  Use --force to re-download")
            return True
    
    print("=" * 60)
    print("SiLens Model Downloader")
    print("=" * 60)
    print()
    print("Model: SmolVLM-256M-Instruct")
    print("Source: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct")
    print("License: Apache 2.0")
    print()
    
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print("ERROR: Required packages not installed.")
        print("Run: pip install transformers huggingface_hub torch")
        sys.exit(1)
    
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    
    print(f"Downloading to: {output_path.absolute()}")
    print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download processor
        print("[1/2] Downloading processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        processor.save_pretrained(output_path)
        print("      ✓ Processor saved")
        
        # Download model
        print("[2/2] Downloading model (this may take a few minutes)...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True
        )
        model.save_pretrained(output_path)
        print("      ✓ Model saved")
        
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("✓ Download complete!")
    print("=" * 60)
    print()
    print(f"Model saved to: {output_path.absolute()}")
    print()
    print("Model components:")
    for f in sorted(output_path.iterdir()):
        size = f.stat().st_size / (1024 * 1024)  # MB
        print(f"  {f.name}: {size:.1f} MB")
    
    return True


def verify_model(model_dir: str = "model/smolvlm-256m"):
    """
    Verify the downloaded model works correctly.
    """
    print()
    print("Verifying model...")
    
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from PIL import Image
        import torch
    except ImportError:
        print("WARNING: Cannot verify - missing dependencies")
        return False
    
    model_path = Path(model_dir)
    
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(model_path)
        
        # Create a simple test image (red square)
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Test inference
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What color is this image?"}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[test_image], return_tensors="pt")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
        
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print("✓ Model verification passed!")
        print(f"  Test output: {output[-100:]}")  # Last 100 chars
        return True
        
    except Exception as e:
        print(f"WARNING: Model verification failed: {e}")
        return False


def print_model_info():
    """
    Print information about SmolVLM-256M.
    """
    info = """
SmolVLM-256M-Instruct
=====================

A compact vision-language model from Hugging Face.

Architecture:
- Vision Encoder: SigLIP-B/16 (93M parameters)
- Multimodal Projector: Linear projection (18M parameters)
- Language Model: SmolLM2-135M (135M parameters)
- Total: ~246M parameters

Capabilities:
- Image captioning
- Visual question answering
- Document understanding
- Multi-image comparison

Input:
- Images: 384x384 pixels (SigLIP encoder)
- Text: Up to 8192 tokens

License: Apache 2.0

More info: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
"""
    print(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SmolVLM-256M model for SiLens project"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model/smolvlm-256m",
        help="Directory to save the model (default: model/smolvlm-256m)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if model already exists"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model after download"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print model information and exit"
    )
    
    args = parser.parse_args()
    
    if args.info:
        print_model_info()
        sys.exit(0)
    
    success = download_model(args.output_dir, args.force)
    
    if success and args.verify:
        verify_model(args.output_dir)
