"""Shared fixtures for tests."""

from datetime import datetime

import pytest

from cc_benchmark.config import (
    BenchmarkConfig,
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelConfig,
)
from cc_benchmark.dataset import SWEBenchInstance
from cc_benchmark.metrics import FailureType, RunRecord


@pytest.fixture
def sample_dataset_config() -> DatasetConfig:
    """Create a sample dataset configuration."""
    return DatasetConfig(
        name="lite",
        split=":5",
        cache_dir="/tmp/swe-bench-test-cache",
    )


@pytest.fixture
def sample_execution_config() -> ExecutionConfig:
    """Create a sample execution configuration."""
    return ExecutionConfig(
        runs=2,
        max_parallel=2,
        timeout_sec=60,
    )


@pytest.fixture
def sample_model_config() -> ModelConfig:
    """Create a sample model configuration."""
    return ModelConfig(
        name="claude-sonnet-4-5",
    )


@pytest.fixture
def sample_benchmark_configs() -> list[BenchmarkConfig]:
    """Create sample benchmark configurations."""
    return [
        BenchmarkConfig(
            name="baseline",
            description="No tools",
            allowed_tools=[],
        ),
        BenchmarkConfig(
            name="with_tools",
            description="Basic tools enabled",
            allowed_tools=["Read", "Write"],
        ),
    ]


@pytest.fixture
def sample_experiment_config(
    sample_dataset_config: DatasetConfig,
    sample_execution_config: ExecutionConfig,
    sample_model_config: ModelConfig,
    sample_benchmark_configs: list[BenchmarkConfig],
) -> ExperimentConfig:
    """Create a sample experiment configuration."""
    return ExperimentConfig(
        name="test-experiment",
        dataset=sample_dataset_config,
        execution=sample_execution_config,
        model=sample_model_config,
        configs=sample_benchmark_configs,
        output_dir="/tmp/test-results",
    )


@pytest.fixture
def sample_instance() -> SWEBenchInstance:
    """Create a sample SWE-bench instance."""
    return SWEBenchInstance(
        instance_id="test__test-123",
        repo="test/test-repo",
        base_commit="abc123",
        problem_statement="Fix the bug in the code",
        test_patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
        FAIL_TO_PASS='["tests/test_example.py::test_bug"]',
        PASS_TO_PASS='["tests/test_example.py::test_other"]',
    )


@pytest.fixture
def sample_run_records() -> list[RunRecord]:
    """Create sample run records for testing aggregation."""
    base_time = datetime.now()
    return [
        RunRecord(
            run_id="run1",
            instance_id="instance1",
            config_id="baseline",
            timestamp=base_time,
            success=True,
            failure_type=FailureType.NONE,
            duration_sec=10.0,
            tokens_input=1000,
            tokens_output=500,
            tokens_cache_read=0,
            tool_calls_total=0,
            cost_usd=0.012,
        ),
        RunRecord(
            run_id="run2",
            instance_id="instance1",
            config_id="baseline",
            timestamp=base_time,
            success=False,
            failure_type=FailureType.TIMEOUT,
            duration_sec=60.0,
            tokens_input=2000,
            tokens_output=1000,
            tokens_cache_read=0,
            tool_calls_total=0,
            cost_usd=0.021,
        ),
        RunRecord(
            run_id="run3",
            instance_id="instance2",
            config_id="with_tools",
            timestamp=base_time,
            success=True,
            failure_type=FailureType.NONE,
            duration_sec=20.0,
            tokens_input=3000,
            tokens_output=1500,
            tokens_cache_read=500,
            tool_calls_total=5,
            tool_calls_by_name={"Read": 3, "Write": 2},
            cost_usd=0.032,
        ),
    ]
