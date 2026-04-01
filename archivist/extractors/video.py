"""Video/audio document extractor using faster-whisper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from archivist.config import Config
from archivist.exceptions import ExtractionError
from archivist.extractors.base import BaseExtractor
from archivist.models import RawDocument


class VideoExtractor(BaseExtractor):
    """Extracts text from video/audio files via Whisper transcription with caching."""

    def __init__(self, config: Config) -> None:
        self._whisper_model = config.whisper.model
        self._cache_dir = Path(config.whisper.cache_dir)

    @property
    def supported_extensions(self) -> list[str]:
        return [".mp4", ".mov", ".mva", ".mp3", ".wav", ".m4a", ".webm"]

    def extract(self, path: Path) -> RawDocument:
        """Transcribe video/audio with caching."""
        cache_path = self._get_cache_path(path)

        if cache_path.exists():
            return self._load_from_cache(path, cache_path)

        return self._transcribe(path, cache_path)

    def _get_cache_path(self, path: Path) -> Path:
        """Generate cache file path based on file hash + model name."""
        file_hash = self._hash_file(path)
        cache_key = f"{file_hash}_{self._whisper_model}"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir / f"{cache_key}.json"

    def _hash_file(self, path: Path) -> str:
        """SHA-256 hash of the file for cache key."""
        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        except Exception as e:
            raise ExtractionError(f"Failed to read file '{path.name}' for hashing: {e}") from e
        return hasher.hexdigest()[:16]

    def _transcribe(self, path: Path, cache_path: Path) -> RawDocument:
        """Run Whisper transcription and cache the result."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ExtractionError(f"faster-whisper is required for video extraction: {e}") from e

        try:
            model = WhisperModel(self._whisper_model)
            raw_segments, _info = model.transcribe(str(path))

            segments = []
            text_parts = []
            for seg in raw_segments:
                segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                })
                text_parts.append(seg.text.strip())

            full_text = " ".join(text_parts)

            # Cache the transcript
            cache_data = {
                "text": full_text,
                "segments": segments,
                "whisper_model": self._whisper_model,
                "source_file": path.name,
            }
            cache_path.write_text(json.dumps(cache_data, indent=2))

            return RawDocument(
                text=full_text,
                source_file=path.name,
                format="video",
                pages=None,
                native_metadata={
                    "duration": segments[-1]["end"] if segments else 0.0,
                    "whisper_model": self._whisper_model,
                    "segment_count": len(segments),
                    "segments": segments,
                },
            )
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(f"Failed to transcribe '{path.name}': {e}") from e

    def _load_from_cache(self, path: Path, cache_path: Path) -> RawDocument:
        """Load transcript from cache."""
        try:
            cache_data = json.loads(cache_path.read_text())
            segments = cache_data.get("segments", [])

            return RawDocument(
                text=cache_data["text"],
                source_file=path.name,
                format="video",
                pages=None,
                native_metadata={
                    "duration": segments[-1]["end"] if segments else 0.0,
                    "whisper_model": cache_data.get("whisper_model", self._whisper_model),
                    "segment_count": len(segments),
                    "segments": segments,
                },
            )
        except Exception as e:
            raise ExtractionError(f"Failed to load cached transcript for '{path.name}': {e}") from e
