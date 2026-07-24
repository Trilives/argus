"""Image loading, resizing, and base64 encoding shared by the model backends."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

import config


def resize_image(image: Image.Image) -> Image.Image:
    """Shrink an image to the configured pixel budget; never upscale."""
    width, height = image.size
    longest = max(width, height)
    pixels = width * height

    scale = 1.0
    if longest > config.IMAGE_LONGEST_EDGE:
        scale = min(scale, config.IMAGE_LONGEST_EDGE / longest)
    if pixels > config.IMAGE_MAX_PIXELS:
        scale = min(scale, (config.IMAGE_MAX_PIXELS / pixels) ** 0.5)
    if scale >= 1.0:
        return image

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.LANCZOS)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return resize_image(image.convert("RGB"))


def encode_image_data_url(image: Image.Image, *, fmt: str = "JPEG") -> str:
    """Encode a PIL image as a ``data:`` URL for OpenAI-style vision messages."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{encoded}"
