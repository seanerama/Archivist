"""Configuration loading and management for Archivist."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from archivist.exceptions import ConfigError


@dataclass
class QdrantConfig:
    """Qdrant connection configuration."""

    host: str = "localhost"
    port: int = 6333
    collection_name: str = "archivist"
    distance_metric: str = "Cosine"
    api_key: str | None = None


@dataclass
class EmbeddingConfig:
    """Embedding backend configuration."""

    type: str = "local"  # "local" or "api"
    # Local options
    model_name: str = "BAAI/bge-m3"
    precision: str = "fp16"  # fp32, fp16, q8
    device: str = "auto"  # auto, cuda, cpu, mps
    batch_size: int = 32
    # API options
    provider: str | None = None  # "voyage"
    model: str | None = None  # e.g. "voyage-3.5"
    api_key: str | None = None


@dataclass
class TaggerConfig:
    """Tagger backend configuration."""

    type: str = "local"  # "local" or "api"
    # Local options
    provider: str = "ollama"
    model: str = "qwen3:0.6b"
    ollama_host: str = "http://localhost:11434"
    # API options
    api_key: str | None = None
    # Tagger behaviour
    auto_accept_tags: bool = False
    auto_accept_threshold: float = 0.90
    new_family_always_review: bool = True


@dataclass
class PipelineConfig:
    """Pipeline behaviour configuration."""

    chunk_size: int = 512
    chunk_overlap_pct: int = 10
    dry_run: bool = False
    overwrite_existing: bool = False


@dataclass
class WhisperConfig:
    """Whisper transcription configuration."""

    model: str = "medium"  # tiny, base, small, medium, large-v3
    cache_dir: str = ".archivist-cache"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"


@dataclass
class Config:
    """Root configuration for Archivist."""

    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    tagger: TaggerConfig = field(default_factory=TaggerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """Load config from a YAML file, resolving env: references."""
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        if raw is None:
            return cls()

        if not isinstance(raw, dict):
            raise ConfigError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")

        raw = _resolve_env_refs(raw)
        return _dict_to_config(raw)

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Load config from YAML file with env var overrides.

        Searches for archivist.yaml in the current directory if no path given.
        Applies ARCHIVIST_ environment variable overrides on top.
        """
        if path is not None:
            config = cls.from_yaml(path)
        elif Path("archivist.yaml").exists():
            config = cls.from_yaml(Path("archivist.yaml"))
        else:
            config = cls()

        _apply_env_overrides(config)
        return config

    @classmethod
    def default(cls) -> Config:
        """Return default configuration."""
        return cls()


def _resolve_env_refs(data: Any) -> Any:
    """Recursively resolve env:VAR_NAME references in config values."""
    if isinstance(data, str) and data.startswith("env:"):
        var_name = data[4:]
        value = os.environ.get(var_name)
        if value is None:
            return None
        return value
    if isinstance(data, dict):
        return {k: _resolve_env_refs(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_refs(item) for item in data]
    return data


def _dict_to_config(raw: dict[str, Any]) -> Config:
    """Convert a raw dict to a Config, ignoring unknown keys."""
    config = Config()

    if "qdrant" in raw and isinstance(raw["qdrant"], dict):
        config.qdrant = _merge_dataclass(QdrantConfig(), raw["qdrant"])

    key = "embedding_backend" if "embedding_backend" in raw else "embedding"
    if key in raw and isinstance(raw[key], dict):
        config.embedding = _merge_dataclass(EmbeddingConfig(), raw[key])

    key = "tagger_backend" if "tagger_backend" in raw else "tagger"
    if key in raw and isinstance(raw[key], dict):
        tagger_data = raw[key]
        config.tagger = _merge_dataclass(TaggerConfig(), tagger_data)

    if "pipeline" in raw and isinstance(raw["pipeline"], dict):
        config.pipeline = _merge_dataclass(PipelineConfig(), raw["pipeline"])

    if "whisper" in raw and isinstance(raw["whisper"], dict):
        config.whisper = _merge_dataclass(WhisperConfig(), raw["whisper"])

    if "logging" in raw and isinstance(raw["logging"], dict):
        config.logging = _merge_dataclass(LoggingConfig(), raw["logging"])

    return config


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> Any:
    """Merge dict values into a dataclass, ignoring unknown keys."""
    for key, value in data.items():
        if hasattr(instance, key) and value is not None:
            setattr(instance, key, value)
    return instance


# Env var override mapping: ARCHIVIST_SECTION_FIELD -> config.section.field
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "ARCHIVIST_QDRANT_HOST": ("qdrant", "host"),
    "ARCHIVIST_QDRANT_PORT": ("qdrant", "port"),
    "ARCHIVIST_QDRANT_COLLECTION_NAME": ("qdrant", "collection_name"),
    "ARCHIVIST_QDRANT_API_KEY": ("qdrant", "api_key"),
    "ARCHIVIST_EMBEDDING_TYPE": ("embedding", "type"),
    "ARCHIVIST_EMBEDDING_MODEL_NAME": ("embedding", "model_name"),
    "ARCHIVIST_EMBEDDING_PRECISION": ("embedding", "precision"),
    "ARCHIVIST_EMBEDDING_DEVICE": ("embedding", "device"),
    "ARCHIVIST_TAGGER_TYPE": ("tagger", "type"),
    "ARCHIVIST_TAGGER_MODEL": ("tagger", "model"),
    "ARCHIVIST_TAGGER_OLLAMA_HOST": ("tagger", "ollama_host"),
    "ARCHIVIST_PIPELINE_CHUNK_SIZE": ("pipeline", "chunk_size"),
    "ARCHIVIST_PIPELINE_DRY_RUN": ("pipeline", "dry_run"),
    "ARCHIVIST_WHISPER_MODEL": ("whisper", "model"),
    "ARCHIVIST_WHISPER_CACHE_DIR": ("whisper", "cache_dir"),
    "ARCHIVIST_LOGGING_LEVEL": ("logging", "level"),
}


def _apply_env_overrides(config: Config) -> None:
    """Apply ARCHIVIST_ environment variable overrides to config."""
    for env_var, (section, field_name) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is None:
            continue

        section_obj = getattr(config, section)
        current = getattr(section_obj, field_name)

        # Type coercion based on the field's current type
        if isinstance(current, bool):
            setattr(section_obj, field_name, value.lower() in ("true", "1", "yes"))
        elif isinstance(current, int):
            with contextlib.suppress(ValueError):
                setattr(section_obj, field_name, int(value))
        elif isinstance(current, float):
            with contextlib.suppress(ValueError):
                setattr(section_obj, field_name, float(value))
        else:
            setattr(section_obj, field_name, value)
