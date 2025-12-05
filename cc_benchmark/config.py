"""Configuration management with Pydantic validation.

This module defines typed configuration models for the benchmark harness,
handles YAML loading, and provides sensible defaults.
"""

from pathlib import Path
from typing import Any

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

    model_config = ConfigDict(extra="forbid", validate_default=True)

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

    model_config = ConfigDict(extra="forbid", validate_default=True)

    runs: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of runs per instance (for pass@k analysis)",
    )
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
        description="Whether to cache Docker images between evaluations (set False to save storage)",
    )
    image_registry: str = Field(
        default="ghcr.io/epoch-research/swe-bench.eval",
        description="Registry prefix for SWE-bench evaluation images",
    )

    @field_validator("image_registry")
    @classmethod
    def validate_image_registry(cls, v: str) -> str:
        """Validate and normalize registry URL."""
        # Remove trailing slash if present
        return v.rstrip("/")


class ModelConfig(BaseModel):
    """Configuration for Claude model parameters.

    Note: The claude-agent-sdk uses Claude Code CLI defaults for max_tokens
    and temperature. These parameters are not configurable through the SDK.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    name: str = Field(
        default="claude-sonnet-4-5",
        description="Claude model identifier",
    )

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is a known Claude model."""
        valid_prefixes = (
            "claude-3",
            "claude-sonnet",
            "claude-opus",
            "claude-haiku",
        )
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"Model name '{v}' does not appear to be a valid Claude model. "
                f"Expected prefix: {valid_prefixes}"
            )
        return v


class Plugin(BaseModel):
    """A Claude Code plugin configuration."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    name: str = Field(description="Plugin display name")
    uri: str = Field(
        description="Local path or GitHub URL (https://github.com/org/repo)"
    )

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, v: str) -> str:
        """Validate URI is a local path or GitHub URL."""
        if v.startswith(("./", "/", "~")):
            return v  # Local path
        if v.startswith("https://github.com/"):
            return v  # GitHub URL
        raise ValueError(
            f"Invalid plugin URI: {v}. "
            "Must be local path (./path, /abs/path, ~/path) or "
            "GitHub URL (https://github.com/org/repo)"
        )


class BenchmarkConfig(BaseModel):
    """Configuration for a single benchmark run."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    name: str = Field(description="Unique identifier for this configuration")
    description: str = Field(default="", description="Description of this configuration")
    plugins: list[Plugin] = Field(
        default_factory=list,
        description="Claude Code plugins to load",
    )
    envs: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for this configuration",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Allowed tools (None = all tools, [] = no tools)",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is a valid identifier."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Config name '{v}' must contain only alphanumeric characters, "
                "underscores, and hyphens"
            )
        return v


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    name: str = Field(description="Experiment name for identification")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    configs: list[BenchmarkConfig] = Field(
        description="List of benchmark configurations to run"
    )
    output_dir: str = Field(
        default="./results",
        description="Directory for benchmark results output",
    )

    @field_validator("configs")
    @classmethod
    def validate_configs(cls, v: list[BenchmarkConfig]) -> list[BenchmarkConfig]:
        """Validate that configs is non-empty and has unique names."""
        if not v:
            raise ValueError("At least one benchmark configuration is required")

        names = [config.name for config in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate config names: {set(duplicates)}")

        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            Validated ExperimentConfig instance

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the file contains invalid YAML
            ValidationError: If the config doesn't match the schema
        """
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Configuration file is empty: {path}")

        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialize configuration to YAML string.

        Returns:
            YAML-formatted string representation
        """
        return yaml.dump(
            self.model_dump(exclude_none=True),
            default_flow_style=False,
            sort_keys=False,
        )

    def get_output_path(self) -> Path:
        """Get resolved output directory path.

        Returns:
            Absolute Path to output directory
        """
        return Path(self.output_dir).expanduser().resolve()
