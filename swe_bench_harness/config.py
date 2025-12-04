"""Configuration management with Pydantic validation.

This module defines typed configuration models for the benchmark harness,
handles YAML loading, and provides sensible defaults.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetConfig(BaseModel):
    """Configuration for SWE-bench dataset loading."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    source: str = Field(
        default="princeton-nlp/SWE-bench_Lite",
        description="HuggingFace dataset source path",
    )
    split: str = Field(
        default="test[:10]",
        description="Dataset split with optional slicing (e.g., 'test[:20]')",
    )
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    cache_dir: str = Field(
        default="~/.cache/swe-bench",
        description="Local cache directory for dataset",
    )


class ExecutionConfig(BaseModel):
    """Configuration for benchmark execution parameters."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    runs_per_instance: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of repeated runs per instance for variance analysis",
    )
    max_parallel_tasks: int = Field(
        default=4,
        ge=1,
        description="Maximum concurrent benchmark executions",
    )
    timeout_per_run_sec: int = Field(
        default=900,
        ge=1,
        description="Timeout in seconds for each run (default: 15 minutes)",
    )


class ModelConfig(BaseModel):
    """Configuration for Claude model parameters."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    name: str = Field(
        default="claude-sonnet-4-5",
        description="Claude model identifier",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=32768,
        description="Maximum output tokens per response",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (lower = more deterministic)",
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


class PluginConfig(BaseModel):
    """Configuration for a single plugin/tool configuration to benchmark."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    id: str = Field(description="Unique identifier for this configuration")
    name: str = Field(description="Human-readable name for display")
    description: str = Field(default="", description="Description of this configuration")
    mcp_servers: dict[str, Any] = Field(
        default_factory=dict,
        description="MCP server configurations {name: config}",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="List of allowed tool names (empty = no tools, ['*'] = all)",
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate ID is a valid identifier."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Plugin ID '{v}' must contain only alphanumeric characters, "
                "underscores, and hyphens"
            )
        return v


class PricingConfig(BaseModel):
    """Configuration for token cost calculation."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    input_cost_per_mtok: float = Field(
        default=3.0,
        ge=0.0,
        description="Cost in USD per million input tokens",
    )
    output_cost_per_mtok: float = Field(
        default=15.0,
        ge=0.0,
        description="Cost in USD per million output tokens",
    )
    cache_read_cost_per_mtok: float = Field(
        default=0.30,
        ge=0.0,
        description="Cost in USD per million cache read tokens",
    )


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    name: str = Field(description="Experiment name for identification")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    configs: list[PluginConfig] = Field(
        description="List of plugin configurations to benchmark"
    )
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    output_dir: str = Field(
        default="./results",
        description="Directory for benchmark results output",
    )

    @field_validator("configs")
    @classmethod
    def validate_configs(cls, v: list[PluginConfig]) -> list[PluginConfig]:
        """Validate that configs is non-empty and has unique IDs."""
        if not v:
            raise ValueError("At least one plugin configuration is required")

        ids = [config.id for config in v]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            raise ValueError(f"Duplicate plugin config IDs: {set(duplicates)}")

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
