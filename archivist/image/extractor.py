"""Image extraction from PDF and EPUB documents."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from archivist.config import ImageConfig
from archivist.log import get_logger
from archivist.models import RawDocument

logger = get_logger("image.extractor")


@dataclass(frozen=True)
class ExtractedImage:
    """An image extracted from a document."""

    data: bytes  # Raw image bytes (PNG)
    source_file: str
    page_number: int | None
    image_index: int
    width: int
    height: int
    caption: str | None


class ImageExtractor:
    """Extracts images from PDF and EPUB documents."""

    def __init__(self, config: ImageConfig) -> None:
        self._config = config
        self._cache_dir = Path(config.cache_dir)

    def extract_images(self, path: Path, raw_doc: RawDocument) -> list[ExtractedImage]:
        """Extract images from a document file.

        Uses a disk cache to avoid re-extracting on subsequent runs.

        Args:
            path: Path to the document file.
            raw_doc: The raw document (used for format detection).

        Returns:
            List of extracted images meeting the minimum size threshold.
        """
        cache_path = self._cache_dir / path.stem
        cached = self._load_from_cache(cache_path, raw_doc.source_file)
        if cached is not None:
            logger.info("Loaded images from cache", file=raw_doc.source_file, count=len(cached))
            return cached

        if raw_doc.format == "pdf":
            images = self._extract_from_pdf(path, raw_doc.source_file)
        elif raw_doc.format == "epub":
            images = self._extract_from_epub(path, raw_doc.source_file)
        else:
            return []

        # Filter by minimum size
        images = [img for img in images if len(img.data) >= self._config.min_size]

        # Cache to disk
        self._save_to_cache(cache_path, images)

        logger.info("Extracted images", file=raw_doc.source_file, count=len(images))
        return images

    def _extract_from_pdf(self, path: Path, source_file: str) -> list[ExtractedImage]:
        """Extract images from a PDF using pymupdf."""
        import fitz

        images: list[ExtractedImage] = []
        doc = fitz.open(str(path))
        image_index = 0

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()

                for img_ref in image_list:
                    xref = img_ref[0]
                    try:
                        extracted = doc.extract_image(xref)
                    except Exception:
                        logger.debug("Failed to extract image", xref=xref, page=page_num)
                        continue

                    if not extracted or "image" not in extracted:
                        continue

                    img_bytes = extracted["image"]
                    width = extracted.get("width", 0)
                    height = extracted.get("height", 0)

                    # Convert to PNG if not already
                    img_bytes = self._ensure_png(img_bytes, extracted.get("ext", ""))

                    images.append(ExtractedImage(
                        data=img_bytes,
                        source_file=source_file,
                        page_number=page_num + 1,  # 1-indexed
                        image_index=image_index,
                        width=width,
                        height=height,
                        caption=None,
                    ))
                    image_index += 1
        finally:
            doc.close()

        return images

    def _extract_from_epub(self, path: Path, source_file: str) -> list[ExtractedImage]:
        """Extract images from an EPUB using ebooklib."""
        import ebooklib
        from ebooklib import epub

        images: list[ExtractedImage] = []
        book = epub.read_epub(str(path))
        image_index = 0

        image_media_types = {
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/gif",
            "image/webp",
            "image/svg+xml",
        }

        for item in book.get_items():
            if item.media_type in image_media_types:
                img_bytes = item.get_content()
                if not img_bytes:
                    continue

                # Try to get dimensions
                width, height = self._get_image_dimensions(img_bytes)

                img_bytes = self._ensure_png(img_bytes, item.media_type.split("/")[-1])

                images.append(ExtractedImage(
                    data=img_bytes,
                    source_file=source_file,
                    page_number=None,
                    image_index=image_index,
                    width=width,
                    height=height,
                    caption=None,
                ))
                image_index += 1

        return images

    @staticmethod
    def _ensure_png(data: bytes, ext: str) -> bytes:
        """Convert image bytes to PNG format if needed."""
        if ext == "png":
            return data
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(data))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            # If PIL isn't available or conversion fails, return as-is
            return data

    @staticmethod
    def _get_image_dimensions(data: bytes) -> tuple[int, int]:
        """Try to determine image dimensions."""
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(data))
            return img.size
        except Exception:
            return (0, 0)

    def _load_from_cache(
        self, cache_path: Path, source_file: str
    ) -> list[ExtractedImage] | None:
        """Load cached images from disk. Returns None if cache miss."""
        if not cache_path.exists():
            return None

        png_files = sorted(cache_path.glob("*.png"))
        if not png_files:
            return None

        images: list[ExtractedImage] = []
        for i, png_file in enumerate(png_files):
            data = png_file.read_bytes()
            width, height = self._get_image_dimensions(data)
            images.append(ExtractedImage(
                data=data,
                source_file=source_file,
                page_number=None,  # Lost in cache
                image_index=i,
                width=width,
                height=height,
                caption=None,
            ))

        return images

    def _save_to_cache(self, cache_path: Path, images: list[ExtractedImage]) -> None:
        """Save extracted images to disk cache."""
        if not images:
            return

        cache_path.mkdir(parents=True, exist_ok=True)
        for img in images:
            filename = f"img_{img.image_index:04d}.png"
            (cache_path / filename).write_bytes(img.data)

        logger.debug("Cached images", path=str(cache_path), count=len(images))
