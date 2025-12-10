"""Configuration management with Pydantic validation.

This module defines typed configuration models for the benchmark harness,
handles YAML loading, and provides sensible defaults.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Dataset name shortcuts - also accepts custom HuggingFace paths
DATASET_SOURCES = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
    "verified": "princeton-nlp/SWE-bench_Verified",
}


class DatasetConfig(BaseModel):
    """Configuration for SWE-bench dataset loading."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="lite",
        description="Dataset name shortcut (lite/full/verified) or HuggingFace path",
    )
    split: str = Field(
        default=":10",
        description="Dataset split slice (e.g., ':10', '20:', '10:20')",
    )
    cache_dir: str = Field(
        default="~/.cc-benchmark",
        description="Local cache directory for dataset",
    )

    @property
    def source(self) -> str:
        """Resolve dataset name to HuggingFace source path."""
        return DATASET_SOURCES.get(self.name, self.name)


class ExecutionConfig(BaseModel):
    """Configuration for benchmark execution parameters."""

    model_config = ConfigDict(extra="forbid")

    max_parallel: int = Field(
        default=4,
        ge=1,
        description="Maximum concurrent benchmark executions",
    )
    timeout_sec: int = Field(
        default=900,
        ge=1,
        description="Timeout in seconds for each run (default: 15 minutes)",
    )
    eval_timeout: int = Field(
        default=1800,
        ge=60,
        description="Timeout in seconds for SWE-bench evaluation (default: 30 minutes)",
    )
    max_turns: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum agent conversation turns (default: 100, was 50)",
    )
    # Docker settings
    docker_image: str = Field(
        default="cc-benchmark-agent:latest",
        description="Docker image for agent execution",
    )
    docker_memory: str = Field(
        default="8g",
        description="Docker container memory limit (e.g., '4g', '8g')",
    )
    docker_cpus: int = Field(
        default=4,
        ge=1,
        description="Docker container CPU limit",
    )
    # Image management
    image_cache: bool = Field(
        default=True,
        description="Whether to cache Docker images between evaluations",
    )
    image_registry: str = Field(
        default="ghcr.io/epoch-research/swe-bench.eval",
        description="Registry prefix for SWE-bench evaluation images",
    )

    @field_validator("image_registry")
    @classmethod
    def validate_image_registry(cls, v: str) -> str:
        """Normalize registry URL."""
        return v.rstrip("/")


class Config(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Experiment name for identification")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    model: str = Field(
        default="claude-sonnet-4-5",
        description="Claude model identifier",
    )
    plugins: list[str] = Field(
        default_factory=list,
        description="Local paths to Claude Code plugins",
    )
    output_dir: str = Field(
        default="./results",
        description="Directory for benchmark results output",
    )

    @field_validator("plugins")
    @classmethod
    def validate_plugins(cls, v: list[str]) -> list[str]:
        """Validate plugin paths are local paths."""
        for path in v:
            if not path.startswith(("./", "/", "~")):
                raise ValueError(
                    f"Invalid plugin path: {path}. "
                    "Must be a local path (./path, /abs/path, ~/path)"
                )
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Configuration file is empty: {path}")

        return cls.model_validate(data)

    def get_output_path(self) -> Path:
        """Get resolved output directory path."""
        return Path(self.output_dir).expanduser().resolve()

    def get_plugin_paths(self) -> list[Path]:
        """Get resolved plugin paths."""
        return [Path(p).expanduser().resolve() for p in self.plugins]
