"""PDF utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # PyMuPDF

if TYPE_CHECKING:
    from PIL import Image


def get_page_count(file_path: Path) -> int:
    """Get the number of pages in a PDF.

    Args:
        file_path: Path to PDF file

    Returns:
        Number of pages
    """
    with fitz.open(file_path) as doc:
        return len(doc)


def get_pdf_metadata(file_path: Path) -> dict:
    """Extract metadata from a PDF.

    Args:
        file_path: Path to PDF file

    Returns:
        Dictionary with title, author, and other metadata
    """
    with fitz.open(file_path) as doc:
        metadata = doc.metadata or {}
        return {
            "title": metadata.get("title") or None,
            "author": metadata.get("author") or None,
            "subject": metadata.get("subject") or None,
            "creator": metadata.get("creator") or None,
            "producer": metadata.get("producer") or None,
            "creation_date": metadata.get("creationDate") or None,
            "modification_date": metadata.get("modDate") or None,
        }


def get_page_text_density(file_path: Path, page_number: int) -> float:
    """Calculate text density for a specific page.

    Text density is the ratio of text area to total page area.
    Low density suggests scanned content that needs OCR.

    Args:
        file_path: Path to PDF file
        page_number: 0-indexed page number

    Returns:
        Float between 0.0 and 1.0 indicating text density
    """
    with fitz.open(file_path) as doc:
        if page_number >= len(doc):
            return 0.0

        page = doc[page_number]
        text = page.get_text("text")

        # If no text at all, density is 0
        if not text.strip():
            return 0.0

        # Calculate approximate text coverage
        # Count non-whitespace characters as a proxy for text presence
        text_chars = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        page_area = page.rect.width * page.rect.height

        # Normalize: assume average 50 chars per 1000 square units is "full" density
        normalized_density = min(1.0, text_chars / (page_area / 20))

        return normalized_density


def is_native_pdf(file_path: Path, threshold: float = 0.1) -> bool:
    """Check if a PDF has native text layer.

    Args:
        file_path: Path to PDF file
        threshold: Minimum average text density to consider native

    Returns:
        True if PDF has sufficient native text
    """
    with fitz.open(file_path) as doc:
        if len(doc) == 0:
            return False

        total_density = 0.0
        for page_num in range(len(doc)):
            total_density += get_page_text_density(file_path, page_num)

        avg_density = total_density / len(doc)
        return avg_density >= threshold


def get_page_classification(file_path: Path, threshold: float = 0.1) -> list[str]:
    """Classify each page as 'native' or 'scanned'.

    Args:
        file_path: Path to PDF file
        threshold: Text density threshold

    Returns:
        List of classifications per page
    """
    classifications = []
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            density = get_page_text_density(file_path, page_num)
            if density >= threshold:
                classifications.append("native")
            else:
                classifications.append("scanned")
    return classifications


def render_page_to_image(
    file_path: Path,
    page_number: int,
    dpi: int = 200,
) -> "Image.Image":
    """Render a PDF page to a PIL Image.

    Args:
        file_path: Path to PDF file
        page_number: 0-indexed page number
        dpi: Resolution for rendering

    Returns:
        PIL Image of the rendered page
    """
    from PIL import Image as PILImage

    with fitz.open(file_path) as doc:
        page = doc[page_number]

        # Calculate zoom factor from DPI (72 is default PDF DPI)
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pixmap = page.get_pixmap(matrix=matrix)

        # Convert to PIL Image
        img = PILImage.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

        return img


def extract_images_from_page(file_path: Path, page_number: int) -> list[dict]:
    """Extract embedded images from a PDF page.

    Args:
        file_path: Path to PDF file
        page_number: 0-indexed page number

    Returns:
        List of dicts with image data and metadata
    """
    from PIL import Image as PILImage

    images = []
    with fitz.open(file_path) as doc:
        page = doc[page_number]
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)

            if base_image:
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Convert to PIL Image
                pil_image = PILImage.open(io.BytesIO(image_bytes))

                images.append({
                    "image": pil_image,
                    "extension": image_ext,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "index": img_index,
                })

    return images


# Need io for BytesIO
import io
