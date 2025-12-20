"""Utility functions for Sibyl.

Provides PDF and image processing utilities.
"""

from sibyl.utils.images import (
    get_image_format,
    image_to_base64,
    is_image_file,
    load_image,
    resize_for_vlm,
)
from sibyl.utils.pdf import (
    extract_images_from_page,
    get_page_classification,
    get_page_count,
    get_page_text_density,
    get_pdf_metadata,
    is_native_pdf,
    render_page_to_image,
)
from sibyl.utils.tables import merge_split_tables

__all__ = [
    # Image utilities
    "resize_for_vlm",
    "image_to_base64",
    "load_image",
    "get_image_format",
    "is_image_file",
    # PDF utilities
    "get_page_count",
    "get_pdf_metadata",
    "get_page_text_density",
    "is_native_pdf",
    "get_page_classification",
    "render_page_to_image",
    "extract_images_from_page",
    # Table utilities
    "merge_split_tables",
]
