"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from swe_bench_harness.config import (
    BenchmarkConfig,
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelConfig,
    Plugin,
)


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_default_values(self):
        """Test that defaults are applied correctly."""
        config = DatasetConfig()
        assert config.name == "lite"
        assert config.source == "princeton-nlp/SWE-bench_Lite"
        assert config.split == ":10"
        assert config.cache_dir == "~/.swe-bench-harness"

    def test_custom_values(self):
        """Test custom value assignment."""
        config = DatasetConfig(
            name="custom/dataset",
            split=":100",
        )
        # Custom name is used as-is (not in DATASET_SOURCES)
        assert config.name == "custom/dataset"
        assert config.source == "custom/dataset"
        assert config.split == ":100"

    def test_shortcut_names(self):
        """Test that shortcut names are resolved correctly."""
        lite_config = DatasetConfig(name="lite")
        assert lite_config.source == "princeton-nlp/SWE-bench_Lite"

        full_config = DatasetConfig(name="full")
        assert full_config.source == "princeton-nlp/SWE-bench"

        verified_config = DatasetConfig(name="verified")
        assert verified_config.source == "princeton-nlp/SWE-bench_Verified"


class TestExecutionConfig:
    """Tests for ExecutionConfig."""

    def test_default_values(self):
        """Test that defaults are applied correctly."""
        config = ExecutionConfig()
        assert config.runs == 1
        assert config.max_parallel == 4
        assert config.timeout_sec == 900

    def test_runs_bounds(self):
        """Test runs validation bounds."""
        # Valid minimum
        config = ExecutionConfig(runs=1)
        assert config.runs == 1

        # Valid maximum
        config = ExecutionConfig(runs=100)
        assert config.runs == 100

        # Below minimum
        with pytest.raises(ValidationError):
            ExecutionConfig(runs=0)

        # Above maximum
        with pytest.raises(ValidationError):
            ExecutionConfig(runs=101)


class TestModelConfig:
    """Tests for ModelConfig.

    Note: max_tokens and temperature were removed as they are not
    supported by the claude-agent-sdk (uses CLI defaults).
    """

    def test_default_values(self):
        """Test that defaults are applied correctly."""
        config = ModelConfig()
        assert config.name == "claude-sonnet-4-5"

    def test_valid_model_names(self):
        """Test that valid model names are accepted."""
        valid_names = [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-haiku-3",
        ]
        for name in valid_names:
            config = ModelConfig(name=name)
            assert config.name == name

    def test_invalid_model_name(self):
        """Test that invalid model names are rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(name="gpt-4")


class TestPlugin:
    """Tests for Plugin configuration."""

    def test_valid_local_path(self):
        """Test valid local path URI."""
        plugin = Plugin(name="test", uri="/path/to/plugin")
        assert plugin.uri == "/path/to/plugin"

    def test_valid_github_url(self):
        """Test valid GitHub URL."""
        plugin = Plugin(name="test", uri="https://github.com/org/repo")
        assert plugin.uri == "https://github.com/org/repo"

    def test_relative_path(self):
        """Test relative path URI."""
        plugin = Plugin(name="test", uri="./local/plugin")
        assert plugin.uri == "./local/plugin"

    def test_home_path(self):
        """Test home directory path."""
        plugin = Plugin(name="test", uri="~/plugins/test")
        assert plugin.uri == "~/plugins/test"

    def test_invalid_uri(self):
        """Test that invalid URIs are rejected."""
        with pytest.raises(ValidationError):
            Plugin(name="test", uri="invalid-uri")


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_valid_config(self):
        """Test valid benchmark configuration."""
        config = BenchmarkConfig(
            name="test_config",
            description="A test config",
            allowed_tools=["Read", "Write"],
        )
        assert config.name == "test_config"
        assert config.description == "A test config"

    def test_name_validation(self):
        """Test that invalid names are rejected."""
        with pytest.raises(ValidationError):
            BenchmarkConfig(name="invalid name", description="Test")  # Space not allowed

    def test_none_allowed_tools(self):
        """Test that None allowed_tools is valid (means all tools)."""
        config = BenchmarkConfig(name="baseline", description="Baseline")
        assert config.allowed_tools is None

    def test_empty_allowed_tools(self):
        """Test that empty allowed_tools is valid."""
        config = BenchmarkConfig(name="baseline", description="Baseline", allowed_tools=[])
        assert config.allowed_tools == []

    def test_with_plugins(self):
        """Test config with plugins."""
        config = BenchmarkConfig(
            name="with_plugins",
            description="Config with plugins",
            plugins=[
                Plugin(name="test", uri="https://github.com/org/repo"),
            ],
        )
        assert len(config.plugins) == 1
        assert config.plugins[0].name == "test"

    def test_with_envs(self):
        """Test config with environment variables."""
        config = BenchmarkConfig(
            name="with_envs",
            description="Config with envs",
            envs={"API_KEY": "test-key"},
        )
        assert config.envs["API_KEY"] == "test-key"


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_valid_config(self, sample_benchmark_configs):
        """Test valid experiment configuration."""
        config = ExperimentConfig(
            name="test",
            configs=sample_benchmark_configs,
        )
        assert config.name == "test"
        assert len(config.configs) == 2

    def test_empty_configs_rejected(self):
        """Test that empty configs list is rejected."""
        with pytest.raises(ValidationError):
            ExperimentConfig(name="test", configs=[])

    def test_duplicate_config_names_rejected(self):
        """Test that duplicate config names are rejected."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                name="test",
                configs=[
                    BenchmarkConfig(name="same", description="First"),
                    BenchmarkConfig(name="same", description="Second"),
                ],
            )

    def test_from_yaml(self, sample_benchmark_configs):
        """Test loading from YAML file."""
        yaml_content = """
name: test-experiment
dataset:
  name: lite
  split: ":5"
execution:
  runs: 2
  max_parallel: 2
model:
  name: claude-sonnet-4-5
configs:
  - name: baseline
    description: Baseline config
    allowed_tools: []
  - name: with_tools
    description: With Tools config
    allowed_tools:
      - Read
output_dir: ./results
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ExperimentConfig.from_yaml(f.name)

            assert config.name == "test-experiment"
            assert config.dataset.split == ":5"
            assert config.execution.runs == 2
            assert len(config.configs) == 2
            assert config.configs[0].name == "baseline"

    def test_from_yaml_missing_file(self):
        """Test that missing YAML file raises error."""
        with pytest.raises(FileNotFoundError):
            ExperimentConfig.from_yaml("/nonexistent/path.yaml")

    def test_to_yaml(self, sample_experiment_config):
        """Test serialization to YAML."""
        yaml_str = sample_experiment_config.to_yaml()
        assert "name: test-experiment" in yaml_str
        assert "baseline" in yaml_str

    def test_get_output_path(self, sample_experiment_config):
        """Test output path resolution."""
        path = sample_experiment_config.get_output_path()
        assert isinstance(path, Path)
        assert path.is_absolute()
