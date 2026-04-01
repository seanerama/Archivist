"""Tests for the video extractor."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from archivist.config import Config
from archivist.exceptions import ExtractionError
from archivist.extractors.video import VideoExtractor


@pytest.fixture
def video_config() -> Config:
    config = Config.default()
    config.whisper.model = "tiny"
    config.whisper.cache_dir = ".test-cache"
    return config


class TestVideoExtractor:
    """Tests for VideoExtractor."""

    def test_supported_extensions(self, video_config: Config) -> None:
        ext = VideoExtractor(video_config)
        assert ".mp4" in ext.supported_extensions
        assert ".mov" in ext.supported_extensions
        assert ".mp3" in ext.supported_extensions

    def test_cache_hit_skips_transcription(self, tmp_path: Path, video_config: Config) -> None:
        video_config.whisper.cache_dir = str(tmp_path / "cache")

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")

        ext = VideoExtractor(video_config)

        # Pre-populate cache
        cache_path = ext._get_cache_path(video_path)
        cache_data = {
            "text": "cached transcript text",
            "segments": [{"start": 0.0, "end": 5.0, "text": "cached transcript text"}],
            "whisper_model": "tiny",
            "source_file": "test.mp4",
        }
        cache_path.write_text(json.dumps(cache_data))

        doc = ext.extract(video_path)
        assert doc.text == "cached transcript text"
        assert doc.format == "video"
        assert doc.native_metadata["segment_count"] == 1

    def test_transcription_and_cache_write(self, tmp_path: Path) -> None:
        config = Config.default()
        config.whisper.model = "tiny"
        config.whisper.cache_dir = str(tmp_path / "cache")

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")

        # Mock WhisperModel
        seg1 = MagicMock(start=0.0, end=5.0, text="Hello world")
        seg2 = MagicMock(start=5.0, end=10.0, text="Second segment")
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = ([seg1, seg2], MagicMock())

        mock_faster_whisper = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model_instance

        orig = sys.modules.get("faster_whisper")
        sys.modules["faster_whisper"] = mock_faster_whisper
        try:
            ext = VideoExtractor(config)
            doc = ext.extract(video_path)

            assert doc.text == "Hello world Second segment"
            assert doc.native_metadata["segment_count"] == 2
            assert doc.native_metadata["duration"] == 10.0

            # Verify cache was written
            cache_path = ext._get_cache_path(video_path)
            assert cache_path.exists()
        finally:
            if orig is not None:
                sys.modules["faster_whisper"] = orig
            else:
                sys.modules.pop("faster_whisper", None)

    def test_missing_file_raises(self, tmp_path: Path, video_config: Config) -> None:
        video_config.whisper.cache_dir = str(tmp_path / "cache")
        ext = VideoExtractor(video_config)
        with pytest.raises(ExtractionError):
            ext.extract(tmp_path / "nonexistent.mp4")
