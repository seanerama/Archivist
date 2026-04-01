"""Tests for configuration loading and management."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from archivist.config import Config
from archivist.exceptions import ConfigError


class TestConfigDefault:
    """Tests for default configuration."""

    def test_default_config_creates_valid_instance(self) -> None:
        config = Config.default()
        assert config.qdrant.host == "localhost"
        assert config.qdrant.port == 6333
        assert config.embedding.type == "local"
        assert config.tagger.type == "local"
        assert config.pipeline.chunk_size == 512
        assert config.whisper.model == "medium"
        assert config.logging.level == "INFO"

    def test_default_config_collection_name(self) -> None:
        config = Config.default()
        assert config.qdrant.collection_name == "archivist"

    def test_default_embedding_model(self) -> None:
        config = Config.default()
        assert config.embedding.model_name == "BAAI/bge-m3"
        assert config.embedding.precision == "fp16"


class TestConfigFromYaml:
    """Tests for YAML config loading."""

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "archivist.yaml"
        config_file.write_text(
            "qdrant:\n  host: myhost\n  port: 7333\npipeline:\n  chunk_size: 1024\n"
        )
        config = Config.from_yaml(config_file)
        assert config.qdrant.host == "myhost"
        assert config.qdrant.port == 7333
        assert config.pipeline.chunk_size == 1024
        # Unspecified values should be defaults
        assert config.embedding.type == "local"

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(ConfigError, match="Config file not found"):
            Config.from_yaml(Path("/nonexistent/archivist.yaml"))

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "archivist.yaml"
        config_file.write_text("")
        config = Config.from_yaml(config_file)
        assert config.qdrant.host == "localhost"

    def test_load_invalid_yaml_type(self, tmp_path: Path) -> None:
        config_file = tmp_path / "archivist.yaml"
        config_file.write_text("just a string")
        with pytest.raises(ConfigError, match="YAML mapping"):
            Config.from_yaml(config_file)

    def test_env_ref_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_QDRANT_KEY", "secret123")
        config_file = tmp_path / "archivist.yaml"
        config_file.write_text("qdrant:\n  api_key: env:TEST_QDRANT_KEY\n")
        config = Config.from_yaml(config_file)
        assert config.qdrant.api_key == "secret123"

    def test_env_ref_missing_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        config_file = tmp_path / "archivist.yaml"
        config_file.write_text("qdrant:\n  api_key: env:NONEXISTENT_VAR\n")
        config = Config.from_yaml(config_file)
        assert config.qdrant.api_key is None

    def test_alternative_section_names(self, tmp_path: Path) -> None:
        """Config supports both 'embedding_backend' and 'embedding' keys."""
        config_file = tmp_path / "archivist.yaml"
        config_file.write_text("embedding_backend:\n  type: api\n  provider: voyage\n")
        config = Config.from_yaml(config_file)
        assert config.embedding.type == "api"
        assert config.embedding.provider == "voyage"


class TestConfigEnvOverrides:
    """Tests for ARCHIVIST_ environment variable overrides."""

    def test_env_override_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARCHIVIST_QDRANT_HOST", "remote-host")
        config = Config.default()
        from archivist.config import _apply_env_overrides

        _apply_env_overrides(config)
        assert config.qdrant.host == "remote-host"

    def test_env_override_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARCHIVIST_QDRANT_PORT", "9999")
        config = Config.default()
        from archivist.config import _apply_env_overrides

        _apply_env_overrides(config)
        assert config.qdrant.port == 9999

    def test_env_override_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARCHIVIST_PIPELINE_DRY_RUN", "true")
        config = Config.default()
        from archivist.config import _apply_env_overrides

        _apply_env_overrides(config)
        assert config.pipeline.dry_run is True

    def test_env_override_does_not_affect_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Ensure no ARCHIVIST_ vars are set
        for key in list(os.environ):
            if key.startswith("ARCHIVIST_"):
                monkeypatch.delenv(key)
        config = Config.default()
        from archivist.config import _apply_env_overrides

        _apply_env_overrides(config)
        assert config.qdrant.host == "localhost"


class TestConfigLoad:
    """Tests for the Config.load() convenience method."""

    def test_load_finds_archivist_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        # Clear any ARCHIVIST_ env vars
        for key in list(os.environ):
            if key.startswith("ARCHIVIST_"):
                monkeypatch.delenv(key)
        config_file = tmp_path / "archivist.yaml"
        config_file.write_text("qdrant:\n  host: from-file\n")
        config = Config.load()
        assert config.qdrant.host == "from-file"

    def test_load_returns_defaults_when_no_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        for key in list(os.environ):
            if key.startswith("ARCHIVIST_"):
                monkeypatch.delenv(key)
        config = Config.load()
        assert config.qdrant.host == "localhost"
