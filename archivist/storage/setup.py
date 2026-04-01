"""Guided Qdrant setup wizard."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt

from archivist.config import Config
from archivist.exceptions import SetupError
from archivist.log import get_logger

logger = get_logger("setup")
console = Console()


class SetupWizard:
    """Interactive setup wizard for Qdrant and backend configuration."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def run(self, config_path: Path = Path("archivist.yaml")) -> Config:
        """Run the interactive setup wizard.

        Args:
            config_path: Path to write the config file.

        Returns:
            Updated Config object.
        """
        console.print("\n[bold]Archivist Setup Wizard[/bold]\n")

        # Step 1: Qdrant
        self._setup_qdrant()

        # Step 2: Validate connection
        self._validate_qdrant()

        # Step 3: Embedding backend
        self._setup_embedding()

        # Step 4: Tagger backend
        self._setup_tagger()

        # Step 5: Write config
        self._write_config(config_path)

        console.print(f"\n[green]Configuration written to {config_path}[/green]")
        return self._config

    def _setup_qdrant(self) -> None:
        """Configure Qdrant connection."""
        console.print("[bold]Step 1: Qdrant Vector Database[/bold]\n")

        # Check if Qdrant is already reachable
        if self._check_qdrant_reachable():
            console.print(
                f"[green]Qdrant is running at "
                f"{self._config.qdrant.host}:{self._config.qdrant.port}[/green]"
            )
            if Confirm.ask("Use this instance?", default=True):
                return

        choice = Prompt.ask(
            "How would you like to run Qdrant?",
            choices=["existing", "docker", "cloud"],
            default="docker",
        )

        if choice == "existing":
            self._config.qdrant.host = Prompt.ask("Qdrant host", default="localhost")
            port_str = Prompt.ask("Qdrant port", default="6333")
            self._config.qdrant.port = int(port_str)

        elif choice == "docker":
            self._provision_docker()

        elif choice == "cloud":
            url = Prompt.ask("Qdrant Cloud URL (e.g. xyz.cloud.qdrant.io)")
            self._config.qdrant.host = url
            self._config.qdrant.port = 6333
            api_key = Prompt.ask("Qdrant API key")
            self._config.qdrant.api_key = api_key

    def _check_qdrant_reachable(self) -> bool:
        """Check if Qdrant is reachable at the configured address."""
        import warnings

        try:
            from qdrant_client import QdrantClient

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                client = QdrantClient(
                    host=self._config.qdrant.host,
                    port=self._config.qdrant.port,
                    timeout=5,
                    check_compatibility=False,
                )
                client.get_collections()
            return True
        except Exception:
            return False

    def _provision_docker(self) -> None:
        """Provision Qdrant via Docker."""
        console.print("Provisioning Qdrant via Docker...")
        try:
            # Check if Docker is available
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            if result.returncode != 0:
                raise SetupError("Docker is not running. Please start Docker and try again.")

            # Check if container already exists
            check = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=archivist-qdrant", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "archivist-qdrant" in check.stdout:
                # Start existing container
                subprocess.run(["docker", "start", "archivist-qdrant"], capture_output=True, timeout=30)
            else:
                # Create new container
                subprocess.run(
                    [
                        "docker", "run", "-d",
                        "--name", "archivist-qdrant",
                        "-p", "6333:6333",
                        "-p", "6334:6334",
                        "-v", "archivist-qdrant-data:/qdrant/storage",
                        "qdrant/qdrant",
                    ],
                    capture_output=True,
                    timeout=60,
                    check=True,
                )

            self._config.qdrant.host = "localhost"
            self._config.qdrant.port = 6333
            console.print("[green]Qdrant Docker container started[/green]")

        except subprocess.CalledProcessError as e:
            raise SetupError(f"Failed to start Qdrant Docker container: {e}") from e
        except FileNotFoundError as e:
            raise SetupError("Docker is not installed. Please install Docker first.") from e
        except subprocess.TimeoutExpired as e:
            raise SetupError("Docker command timed out.") from e

    def _validate_qdrant(self) -> None:
        """Validate the Qdrant connection, retrying for newly started containers."""
        import time

        console.print("\nValidating Qdrant connection...")
        for attempt in range(6):
            if self._check_qdrant_reachable():
                console.print("[green]Connection successful![/green]")
                return
            if attempt < 5:
                console.print(f"  Waiting for Qdrant to start... ({attempt + 1}/5)")
                time.sleep(2)

        console.print(
            f"[red]Cannot reach Qdrant at "
            f"{self._config.qdrant.host}:{self._config.qdrant.port}[/red]"
        )
        console.print("Please check your configuration and try again.")

    def _setup_embedding(self) -> None:
        """Configure the embedding backend."""
        console.print("\n[bold]Step 2: Embedding Backend[/bold]\n")

        choice = Prompt.ask(
            "Embedding backend",
            choices=["local", "api"],
            default="local",
        )
        self._config.embedding.type = choice

        if choice == "local":
            model = Prompt.ask("HuggingFace model", default="BAAI/bge-m3")
            self._config.embedding.model_name = model

            precision = Prompt.ask("Precision", choices=["fp32", "fp16", "q8"], default="fp16")
            self._config.embedding.precision = precision

            device = Prompt.ask("Device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
            self._config.embedding.device = device
        else:
            provider = Prompt.ask("API provider", choices=["voyage"], default="voyage")
            self._config.embedding.provider = provider

            model = Prompt.ask("Model name", default="voyage-3.5")
            self._config.embedding.model = model

            console.print("Set your API key in .env: [bold]VOYAGE_API_KEY=your-key[/bold]")
            self._config.embedding.api_key = "env:VOYAGE_API_KEY"

    def _setup_tagger(self) -> None:
        """Configure the tagger backend."""
        console.print("\n[bold]Step 3: Document Tagger Backend[/bold]\n")

        choice = Prompt.ask(
            "Tagger backend",
            choices=["local", "api"],
            default="local",
        )
        self._config.tagger.type = choice

        if choice == "local":
            self._config.tagger.provider = "ollama"
            model = Prompt.ask("Ollama model", default="qwen3:0.6b")
            self._config.tagger.model = model

            host = Prompt.ask("Ollama host", default="http://localhost:11434")
            self._config.tagger.ollama_host = host

            console.print(f"Make sure the model is pulled: [bold]ollama pull {model}[/bold]")
        else:
            self._config.tagger.provider = "anthropic"
            model = Prompt.ask("Model", default="claude-haiku-4-5-20251001")
            self._config.tagger.model = model

            console.print("Set your API key in .env: [bold]ANTHROPIC_API_KEY=your-key[/bold]")
            self._config.tagger.api_key = "env:ANTHROPIC_API_KEY"

    def _write_config(self, config_path: Path) -> None:
        """Write current config to YAML file."""
        config_dict: dict[str, Any] = {
            "qdrant": {
                "host": self._config.qdrant.host,
                "port": self._config.qdrant.port,
                "collection_name": self._config.qdrant.collection_name,
                "distance_metric": self._config.qdrant.distance_metric,
            },
            "embedding_backend": {
                "type": self._config.embedding.type,
                "model_name": self._config.embedding.model_name,
                "precision": self._config.embedding.precision,
                "device": self._config.embedding.device,
                "batch_size": self._config.embedding.batch_size,
            },
            "tagger_backend": {
                "type": self._config.tagger.type,
                "provider": self._config.tagger.provider,
                "model": self._config.tagger.model,
                "auto_accept_tags": self._config.tagger.auto_accept_tags,
                "auto_accept_threshold": self._config.tagger.auto_accept_threshold,
                "new_family_always_review": self._config.tagger.new_family_always_review,
            },
            "pipeline": {
                "chunk_size": self._config.pipeline.chunk_size,
                "chunk_overlap_pct": self._config.pipeline.chunk_overlap_pct,
            },
            "whisper": {
                "model": self._config.whisper.model,
                "cache_dir": self._config.whisper.cache_dir,
            },
            "logging": {
                "level": self._config.logging.level,
            },
        }

        if self._config.qdrant.api_key:
            config_dict["qdrant"]["api_key"] = "env:QDRANT_API_KEY"

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
