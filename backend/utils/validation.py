"""
Input and Image Upload Validation Utilities.

Enforces security and integrity checks on all uploaded files:
- Never trusts user-supplied filenames; sanitizes with `secure_filename` and
  replaces storage names with server-generated UUIDs.
- Validates allowed extensions (.jpg, .jpeg, .png).
- Validates non-empty payload and maximum byte size.
- Verifies image structure with Pillow (`Image.verify()` and `Image.load()`).
- Enforces minimum (32x32) and maximum (8192x8192) pixel dimensions.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from typing import Tuple

from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from backend.config import (
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    MAX_IMAGE_DIMENSION,
    MIN_IMAGE_DIMENSION,
)


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails security, format, or dimension checks."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class ValidatedUpload:
    """Container for a validated image upload."""

    safe_original_filename: str
    unique_id: str
    extension: str
    raw_bytes: bytes
    pil_image: Image.Image
    width: int
    height: int


def is_allowed_filename(filename: str) -> Tuple[bool, str]:
    """Check whether the filename has a valid extension."""
    if not filename or "." not in filename:
        return False, ""
    ext = filename.rsplit(".", 1)[1].lower().strip()
    return (ext in ALLOWED_EXTENSIONS), ext


def validate_uploaded_image(file_storage: FileStorage | None) -> ValidatedUpload:
    """
    Validate a Flask `FileStorage` upload object and return a `ValidatedUpload`
    containing the verified RGB PIL Image and sanitized metadata.
    """
    if file_storage is None or not file_storage.filename:
        raise ImageValidationError("No image file was provided in the request.", status_code=400)

    safe_name = secure_filename(file_storage.filename)
    allowed, ext = is_allowed_filename(safe_name)
    if not allowed:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ImageValidationError(
            f"Unsupported file type '{ext or 'unknown'}'. Allowed formats: {allowed_list}.",
            status_code=400,
        )

    raw_bytes = file_storage.read()
    if not raw_bytes:
        raise ImageValidationError("Uploaded image file is empty (0 bytes).", status_code=400)

    if len(raw_bytes) > MAX_CONTENT_LENGTH:
        max_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
        raise ImageValidationError(
            f"Uploaded file exceeds maximum allowed size of {max_mb:.1f} MB.",
            status_code=413,
        )

    try:
        verify_stream = io.BytesIO(raw_bytes)
        with Image.open(verify_stream) as img_check:
            img_check.verify()

        load_stream = io.BytesIO(raw_bytes)
        with Image.open(load_stream) as img_loaded:
            img_loaded.load()
            width, height = img_loaded.size
            rgb_image = img_loaded.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(
            f"Corrupted or unreadable image file. Please upload a valid JPG or PNG image. ({exc})",
            status_code=400,
        ) from exc

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) are too small. "
            f"Minimum supported size is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels.",
            status_code=400,
        )

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) exceed maximum supported size "
            f"({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels).",
            status_code=400,
        )

    unique_id = f"pred_{uuid.uuid4().hex[:12]}"
    return ValidatedUpload(
        safe_original_filename=safe_name,
        unique_id=unique_id,
        extension=ext,
        raw_bytes=raw_bytes,
        pil_image=rgb_image,
        width=width,
        height=height,
    )
