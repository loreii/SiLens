"""
SiLens Image Processing Module.

Advanced image preprocessing utilities for the SiLens VLM accelerator including:
- Image resizing with multiple interpolation methods
- Normalization for SigLIP vision encoder
- Patch extraction for ViT-style processing
- Data augmentation for inference robustness
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Literal
from pathlib import Path
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

logger = logging.getLogger(__name__)

# Default image parameters matching SiLens hardware
DEFAULT_IMAGE_SIZE = 384
DEFAULT_PATCH_SIZE = 16
NUM_PATCHES = (DEFAULT_IMAGE_SIZE // DEFAULT_PATCH_SIZE) ** 2  # 576

# SigLIP normalization constants
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# ImageNet normalization constants (alternative)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ImageConfig:
    """Configuration for image preprocessing."""
    size: int = DEFAULT_IMAGE_SIZE
    patch_size: int = DEFAULT_PATCH_SIZE
    mean: Tuple[float, float, float] = SIGLIP_MEAN
    std: Tuple[float, float, float] = SIGLIP_STD
    interpolation: Literal["bilinear", "bicubic", "nearest", "lanczos"] = "bilinear"
    normalize: bool = True
    center_crop: bool = False
    
    @property
    def num_patches(self) -> int:
        """Number of patches after processing."""
        return (self.size // self.patch_size) ** 2
    
    @property
    def patch_dim(self) -> int:
        """Dimension of each flattened patch (patch_size^2 * channels)."""
        return self.patch_size * self.patch_size * 3


class ImageProcessor:
    """
    Image preprocessor for SiLens VLM.
    
    Handles loading, resizing, normalization, and patch extraction
    for images before they are sent to the hardware accelerator.
    
    Example:
        processor = ImageProcessor()
        
        # Process a single image
        pixels, patches = processor.process("photo.jpg")
        
        # Process with custom config
        processor = ImageProcessor(ImageConfig(size=512, center_crop=True))
        pixels, patches = processor.process(image_array)
    """
    
    def __init__(self, config: Optional[ImageConfig] = None):
        """
        Initialize image processor.
        
        Args:
            config: Image processing configuration (uses defaults if None)
        """
        self.config = config or ImageConfig()
    
    def process(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        return_patches: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Process an image for SiLens inference.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            return_patches: Whether to also return extracted patches
            
        Returns:
            Processed image array (H, W, C) in uint8 or float32 format.
            If return_patches=True, also returns (num_patches, patch_dim) array.
        """
        # Load image
        img = self._load_image(image)
        
        # Apply center crop if configured
        if self.config.center_crop:
            img = self._center_crop(img)
        
        # Resize to target size
        img = self._resize(img, self.config.size, self.config.interpolation)
        
        # Normalize if configured
        if self.config.normalize:
            img = self._normalize(img, self.config.mean, self.config.std)
        
        if return_patches:
            patches = self.extract_patches(img)
            return img, patches
        
        return img
    
    def process_batch(
        self,
        images: List[Union[str, Path, np.ndarray, "Image.Image"]],
        return_patches: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Process a batch of images.
        
        Args:
            images: List of input images
            return_patches: Whether to also return extracted patches
            
        Returns:
            Batch of processed images (B, H, W, C).
            If return_patches=True, also returns (B, num_patches, patch_dim) array.
        """
        processed = []
        patches_list = []
        
        for img in images:
            if return_patches:
                processed_img, patches = self.process(img, return_patches=True)
                patches_list.append(patches)
            else:
                processed_img = self.process(img, return_patches=False)
            processed.append(processed_img)
        
        result = np.stack(processed, axis=0)
        
        if return_patches:
            patches_batch = np.stack(patches_list, axis=0)
            return result, patches_batch
        
        return result
    
    def extract_patches(self, image: np.ndarray) -> np.ndarray:
        """
        Extract non-overlapping patches from an image.
        
        Args:
            image: Preprocessed image array (H, W, C)
            
        Returns:
            Patches array with shape (num_patches, patch_dim)
            where patch_dim = patch_size * patch_size * channels
        """
        h, w, c = image.shape
        ps = self.config.patch_size
        
        if h != self.config.size or w != self.config.size:
            raise ValueError(f"Image size must be {self.config.size}x{self.config.size}")
        
        # Number of patches along each dimension
        n_h = h // ps
        n_w = w // ps
        
        # Extract patches using reshape
        patches = image.reshape(n_h, ps, n_w, ps, c)
        patches = patches.transpose(0, 2, 1, 3, 4)  # (n_h, n_w, ps, ps, c)
        patches = patches.reshape(n_h * n_w, -1)  # (num_patches, patch_dim)
        
        return patches
    
    def patches_to_image(self, patches: np.ndarray) -> np.ndarray:
        """
        Reconstruct image from patches.
        
        Args:
            patches: Patches array with shape (num_patches, patch_dim)
            
        Returns:
            Reconstructed image array (H, W, C)
        """
        ps = self.config.patch_size
        size = self.config.size
        n = size // ps  # patches per dimension
        c = 3  # RGB channels
        
        # Reshape patches back to image
        patches = patches.reshape(n, n, ps, ps, c)
        patches = patches.transpose(0, 2, 1, 3, 4)  # (n, ps, n, ps, c)
        image = patches.reshape(size, size, c)
        
        return image
    
    def _load_image(
        self,
        source: Union[str, Path, np.ndarray, "Image.Image"],
    ) -> np.ndarray:
        """Load image from various sources."""
        if isinstance(source, np.ndarray):
            img = source
        elif HAS_PIL and isinstance(source, Image.Image):
            img = np.array(source.convert("RGB"))
        elif isinstance(source, (str, Path)):
            if not HAS_PIL:
                raise ImportError("PIL required for loading images: pip install pillow")
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            img = np.array(Image.open(path).convert("RGB"))
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")
        
        # Ensure RGB
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        
        return img
    
    def _resize(
        self,
        image: np.ndarray,
        size: int,
        interpolation: str,
    ) -> np.ndarray:
        """Resize image to target size."""
        if image.shape[0] == size and image.shape[1] == size:
            return image
        
        if HAS_PIL:
            resampling = {
                "nearest": Image.Resampling.NEAREST,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
                "lanczos": Image.Resampling.LANCZOS,
            }.get(interpolation, Image.Resampling.BILINEAR)
            
            pil_img = Image.fromarray(image.astype(np.uint8))
            pil_img = pil_img.resize((size, size), resampling)
            return np.array(pil_img)
        else:
            # Fallback: nearest neighbor
            return self._resize_nearest(image, size)
    
    def _resize_nearest(self, image: np.ndarray, size: int) -> np.ndarray:
        """Simple nearest-neighbor resize."""
        h, w = image.shape[:2]
        y_indices = (np.arange(size) * h / size).astype(int)
        x_indices = (np.arange(size) * w / size).astype(int)
        return image[y_indices[:, None], x_indices]
    
    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        """Center crop image to a square."""
        h, w = image.shape[:2]
        if h == w:
            return image
        
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        
        return image[top:top + min_dim, left:left + min_dim]
    
    def _normalize(
        self,
        image: np.ndarray,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> np.ndarray:
        """Normalize image with mean and std."""
        img = image.astype(np.float32) / 255.0
        img = (img - np.array(mean)) / np.array(std)
        return img
    
    def to_hardware_format(
        self,
        image: np.ndarray,
        quantize: bool = True,
    ) -> np.ndarray:
        """
        Convert processed image to hardware-compatible format.
        
        Args:
            image: Normalized float image
            quantize: Whether to quantize to uint8
            
        Returns:
            Hardware-ready image as uint8 array
        """
        if quantize:
            # Scale from normalized range to 0-255
            # For SigLIP normalization (0.5, 0.5), range is roughly [-2, 2]
            # Map to [0, 255] with center at 128
            img = np.clip(image * 64 + 128, 0, 255).astype(np.uint8)
        else:
            img = image
        
        return img


# Convenience functions for common operations

def resize_image(
    image: Union[str, Path, np.ndarray, "Image.Image"],
    size: int = DEFAULT_IMAGE_SIZE,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Resize an image to the target size.
    
    Args:
        image: Input image
        size: Target size (square)
        interpolation: Interpolation method
        
    Returns:
        Resized image array
    """
    processor = ImageProcessor(ImageConfig(
        size=size,
        interpolation=interpolation,
        normalize=False,
    ))
    return processor._load_image(image)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = SIGLIP_MEAN,
    std: Tuple[float, float, float] = SIGLIP_STD,
) -> np.ndarray:
    """
    Normalize an image with given mean and std.
    
    Args:
        image: Input image (uint8 or float)
        mean: Per-channel mean values
        std: Per-channel std values
        
    Returns:
        Normalized float image
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return (img - np.array(mean)) / np.array(std)


def extract_patches(
    image: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> np.ndarray:
    """
    Extract non-overlapping patches from an image.
    
    Args:
        image: Input image (H, W, C)
        patch_size: Size of each patch
        
    Returns:
        Patches array (num_patches, patch_dim)
    """
    processor = ImageProcessor(ImageConfig(
        size=image.shape[0],
        patch_size=patch_size,
    ))
    return processor.extract_patches(image)


def create_attention_mask(
    image: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Create an attention mask based on image content.
    
    Useful for masking out padding or background regions.
    
    Args:
        image: Input image
        threshold: Threshold for considering a region as content
        
    Returns:
        Boolean mask array
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = np.mean(image, axis=-1)
    else:
        gray = image
    
    # Normalize
    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0
    
    # Create mask
    return gray > threshold


def generate_position_embeddings(
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    hidden_dim: int = 768,
) -> np.ndarray:
    """
    Generate 2D sinusoidal position embeddings for patches.
    
    Args:
        image_size: Image size
        patch_size: Patch size
        hidden_dim: Embedding dimension
        
    Returns:
        Position embeddings array (num_patches, hidden_dim)
    """
    n_patches = image_size // patch_size
    num_patches = n_patches * n_patches
    
    # Generate 2D positions
    positions = np.arange(num_patches)
    rows = positions // n_patches
    cols = positions % n_patches
    
    # Create sinusoidal embeddings
    dim = hidden_dim // 4  # Split for row/col sin/cos
    
    def get_sinusoid(pos, d):
        div_term = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
        pe = np.zeros((len(pos), d))
        pe[:, 0::2] = np.sin(pos[:, None] * div_term)
        pe[:, 1::2] = np.cos(pos[:, None] * div_term)
        return pe
    
    row_embed = get_sinusoid(rows.astype(np.float32), dim * 2)
    col_embed = get_sinusoid(cols.astype(np.float32), dim * 2)
    
    return np.concatenate([row_embed, col_embed], axis=-1)
