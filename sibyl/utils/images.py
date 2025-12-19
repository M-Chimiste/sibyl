"""Image utility functions."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def resize_for_vlm(
    image: "Image.Image",
    max_dimension: int = 2048,
) -> "Image.Image":
    """Resize image to fit within VLM context limits.

    Maintains aspect ratio while ensuring neither dimension
    exceeds max_dimension.

    Args:
        image: PIL Image to resize
        max_dimension: Maximum width or height

    Returns:
        Resized PIL Image (or original if already small enough)
    """
    width, height = image.size

    if width <= max_dimension and height <= max_dimension:
        return image

    # Calculate scale factor
    scale = min(max_dimension / width, max_dimension / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Use high-quality resampling
    from PIL import Image as PILImage

    return image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)


def image_to_base64(
    image: "Image.Image",
    format: str = "PNG",
) -> str:
    """Convert PIL Image to base64 string.

    Args:
        image: PIL Image to convert
        format: Output format (PNG, JPEG, etc.)

    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()

    # Convert RGBA to RGB for JPEG
    if format.upper() == "JPEG" and image.mode == "RGBA":
        image = image.convert("RGB")

    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_image(file_path: Path) -> "Image.Image":
    """Load an image from file.

    Args:
        file_path: Path to image file

    Returns:
        PIL Image
    """
    from PIL import Image as PILImage

    return PILImage.open(file_path)


def get_image_format(file_path: Path) -> str:
    """Detect image format from file.

    Args:
        file_path: Path to image file

    Returns:
        Format string (PNG, JPEG, TIFF, etc.)
    """
    from PIL import Image as PILImage

    with PILImage.open(file_path) as img:
        return img.format or "PNG"


def is_image_file(file_path: Path) -> bool:
    """Check if a file is a supported image format.

    Args:
        file_path: Path to check

    Returns:
        True if file is a supported image
    """
    supported_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
    return file_path.suffix.lower() in supported_extensions


def estimate_image_tokens(image: "Image.Image") -> int:
    """Estimate token count for an image in VLM context.

    This is a rough estimate based on image dimensions.
    Different VLMs have different tokenization strategies.

    Args:
        image: PIL Image

    Returns:
        Estimated token count
    """
    width, height = image.size
    # Rough estimate: ~1 token per 14x14 pixel patch (common in vision transformers)
    patches = (width // 14) * (height // 14)
    # Add overhead for image tokens
    return patches + 100
