"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from swe_bench_harness.config import (
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelConfig,
    PluginConfig,
    PricingConfig,
)


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_default_values(self):
        """Test that defaults are applied correctly."""
        config = DatasetConfig()
        assert config.source == "princeton-nlp/SWE-bench_Lite"
        assert config.split == "test[:10]"
        assert config.seed == 42
        assert config.cache_dir == "~/.cache/swe-bench"

    def test_custom_values(self):
        """Test custom value assignment."""
        config = DatasetConfig(
            source="custom/dataset",
            split="train[:100]",
            seed=123,
        )
        assert config.source == "custom/dataset"
        assert config.split == "train[:100]"
        assert config.seed == 123

    def test_seed_validation(self):
        """Test that negative seed is rejected."""
        with pytest.raises(ValidationError):
            DatasetConfig(seed=-1)


class TestExecutionConfig:
    """Tests for ExecutionConfig."""

    def test_default_values(self):
        """Test that defaults are applied correctly."""
        config = ExecutionConfig()
        assert config.runs_per_instance == 5
        assert config.max_parallel_tasks == 4
        assert config.timeout_per_run_sec == 900

    def test_runs_per_instance_bounds(self):
        """Test runs_per_instance validation bounds."""
        # Valid minimum
        config = ExecutionConfig(runs_per_instance=1)
        assert config.runs_per_instance == 1

        # Valid maximum
        config = ExecutionConfig(runs_per_instance=100)
        assert config.runs_per_instance == 100

        # Below minimum
        with pytest.raises(ValidationError):
            ExecutionConfig(runs_per_instance=0)

        # Above maximum
        with pytest.raises(ValidationError):
            ExecutionConfig(runs_per_instance=101)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test that defaults are applied correctly."""
        config = ModelConfig()
        assert config.name == "claude-sonnet-4-5"
        assert config.max_tokens == 4096
        assert config.temperature == 0.2

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

    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        # Valid range
        config = ModelConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = ModelConfig(temperature=1.0)
        assert config.temperature == 1.0

        # Below minimum
        with pytest.raises(ValidationError):
            ModelConfig(temperature=-0.1)

        # Above maximum
        with pytest.raises(ValidationError):
            ModelConfig(temperature=1.1)


class TestPluginConfig:
    """Tests for PluginConfig."""

    def test_valid_config(self):
        """Test valid plugin configuration."""
        config = PluginConfig(
            id="test_plugin",
            name="Test Plugin",
            description="A test plugin",
            allowed_tools=["read_file", "write_file"],
        )
        assert config.id == "test_plugin"
        assert config.name == "Test Plugin"

    def test_id_validation(self):
        """Test that invalid IDs are rejected."""
        with pytest.raises(ValidationError):
            PluginConfig(id="invalid id", name="Test")  # Space not allowed

    def test_empty_allowed_tools(self):
        """Test that empty allowed_tools is valid."""
        config = PluginConfig(id="baseline", name="Baseline")
        assert config.allowed_tools == []


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_valid_config(self, sample_plugin_configs):
        """Test valid experiment configuration."""
        config = ExperimentConfig(
            name="test",
            configs=sample_plugin_configs,
        )
        assert config.name == "test"
        assert len(config.configs) == 2

    def test_empty_configs_rejected(self):
        """Test that empty configs list is rejected."""
        with pytest.raises(ValidationError):
            ExperimentConfig(name="test", configs=[])

    def test_duplicate_config_ids_rejected(self):
        """Test that duplicate config IDs are rejected."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                name="test",
                configs=[
                    PluginConfig(id="same", name="First"),
                    PluginConfig(id="same", name="Second"),
                ],
            )

    def test_from_yaml(self, sample_plugin_configs):
        """Test loading from YAML file."""
        yaml_content = """
name: test-experiment
dataset:
  source: princeton-nlp/SWE-bench_Lite
  split: test[:5]
execution:
  runs_per_instance: 2
  max_parallel_tasks: 2
model:
  name: claude-sonnet-4-5
configs:
  - id: baseline
    name: Baseline
    allowed_tools: []
  - id: with_tools
    name: With Tools
    allowed_tools:
      - read_file
output_dir: ./results
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ExperimentConfig.from_yaml(f.name)

            assert config.name == "test-experiment"
            assert config.dataset.split == "test[:5]"
            assert config.execution.runs_per_instance == 2
            assert len(config.configs) == 2
            assert config.configs[0].id == "baseline"

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
