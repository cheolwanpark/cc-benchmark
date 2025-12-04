"""Tests for metrics collection and aggregation."""

from datetime import datetime

import pytest

from swe_bench_harness.config import PluginConfig, PricingConfig
from swe_bench_harness.metrics import (
    BenchmarkResults,
    ConfigSummary,
    FailureType,
    MetricsAggregator,
    RunRecord,
)


class TestRunRecord:
    """Tests for RunRecord dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        record = RunRecord(
            run_id="test_run",
            instance_id="instance_1",
            config_id="config_1",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            success=True,
            failure_type=FailureType.NONE,
            duration_sec=10.5,
            tokens_input=1000,
            tokens_output=500,
            tokens_cache_read=100,
            tool_calls_total=3,
            tool_calls_by_name={"read_file": 2, "write_file": 1},
            cost_usd=0.015,
        )

        data = record.to_dict()

        assert data["run_id"] == "test_run"
        assert data["instance_id"] == "instance_1"
        assert data["success"] is True
        assert data["failure_type"] == "none"
        assert data["tokens_input"] == 1000
        assert data["tool_calls_by_name"] == {"read_file": 2, "write_file": 1}

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "run_id": "test_run",
            "instance_id": "instance_1",
            "config_id": "config_1",
            "timestamp": "2024-01-01T12:00:00",
            "success": True,
            "failure_type": "none",
            "duration_sec": 10.5,
            "tokens_input": 1000,
            "tokens_output": 500,
            "tokens_cache_read": 100,
            "tool_calls_total": 3,
            "tool_calls_by_name": {"read_file": 2, "write_file": 1},
            "cost_usd": 0.015,
        }

        record = RunRecord.from_dict(data)

        assert record.run_id == "test_run"
        assert record.success is True
        assert record.failure_type == FailureType.NONE
        assert record.tokens_input == 1000

    def test_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        original = RunRecord(
            run_id="test",
            instance_id="inst",
            config_id="cfg",
            timestamp=datetime.now(),
            success=False,
            failure_type=FailureType.TIMEOUT,
            error_reason="Timed out",
        )

        data = original.to_dict()
        restored = RunRecord.from_dict(data)

        assert restored.run_id == original.run_id
        assert restored.failure_type == original.failure_type
        assert restored.error_reason == original.error_reason


class TestMetricsAggregator:
    """Tests for MetricsAggregator."""

    @pytest.fixture
    def aggregator(self) -> MetricsAggregator:
        """Create an aggregator with default pricing."""
        return MetricsAggregator(PricingConfig())

    @pytest.fixture
    def configs(self) -> list[PluginConfig]:
        """Create sample plugin configs."""
        return [
            PluginConfig(id="baseline", name="Baseline"),
            PluginConfig(id="with_tools", name="With Tools"),
        ]

    def test_calculate_cost(self, aggregator):
        """Test cost calculation."""
        cost = aggregator.calculate_cost(
            tokens_input=1_000_000,
            tokens_output=1_000_000,
            tokens_cache_read=0,
        )
        # 1M input @ $3 + 1M output @ $15 = $18
        assert cost == pytest.approx(18.0)

    def test_calculate_cost_with_cache(self, aggregator):
        """Test cost calculation with cache reads."""
        cost = aggregator.calculate_cost(
            tokens_input=1_000_000,
            tokens_output=500_000,
            tokens_cache_read=1_000_000,
        )
        # 1M input @ $3 + 0.5M output @ $15 + 1M cache @ $0.30 = $10.80
        assert cost == pytest.approx(10.80)

    def test_aggregate_empty_records(self, aggregator, configs):
        """Test aggregation with no records."""
        summaries = aggregator.aggregate([], configs)
        assert summaries == []

    def test_aggregate_single_record(self, aggregator, configs):
        """Test aggregation with single record."""
        records = [
            RunRecord(
                run_id="run1",
                instance_id="inst1",
                config_id="baseline",
                timestamp=datetime.now(),
                success=True,
                duration_sec=10.0,
                tokens_input=1000,
                tokens_output=500,
                tool_calls_total=0,
                cost_usd=0.01,
            )
        ]

        summaries = aggregator.aggregate(records, configs)

        assert len(summaries) == 1
        assert summaries[0].config_id == "baseline"
        assert summaries[0].total_runs == 1
        assert summaries[0].success_count == 1
        assert summaries[0].success_rate == 1.0
        assert summaries[0].duration_mean == 10.0
        assert summaries[0].tokens_mean == 1500

    def test_aggregate_multiple_configs(self, aggregator, sample_run_records, configs):
        """Test aggregation with multiple configurations."""
        summaries = aggregator.aggregate(sample_run_records, configs)

        # Should have 2 configs
        assert len(summaries) == 2

        # Check baseline
        baseline = next(s for s in summaries if s.config_id == "baseline")
        assert baseline.total_runs == 2
        assert baseline.success_count == 1
        assert baseline.success_rate == 0.5

        # Check with_tools
        with_tools = next(s for s in summaries if s.config_id == "with_tools")
        assert with_tools.total_runs == 1
        assert with_tools.success_count == 1
        assert with_tools.success_rate == 1.0

    def test_percentile_calculation(self, aggregator):
        """Test percentile calculation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        p50 = aggregator._percentile(data, 50)
        assert p50 == pytest.approx(5.5)

        p90 = aggregator._percentile(data, 90)
        assert p90 == pytest.approx(9.1)

    def test_percentile_single_value(self, aggregator):
        """Test percentile with single value."""
        p50 = aggregator._percentile([5.0], 50)
        assert p50 == 5.0

    def test_percentile_empty(self, aggregator):
        """Test percentile with empty list."""
        p50 = aggregator._percentile([], 50)
        assert p50 == 0.0


class TestConfigSummary:
    """Tests for ConfigSummary."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        summary = ConfigSummary(
            config_id="test",
            config_name="Test Config",
            total_runs=10,
            success_count=7,
            success_rate=0.7,
            duration_mean=15.5,
            duration_p50=14.0,
            duration_p90=20.0,
            duration_std=3.2,
            tokens_mean=1500,
            tool_calls_mean=2.5,
            cost_mean=0.02,
            cost_total=0.2,
        )

        data = summary.to_dict()

        assert data["config_id"] == "test"
        assert data["success_rate"] == 0.7
        assert data["duration_mean"] == 15.5
        assert data["tokens_mean"] == 1500


class TestBenchmarkResults:
    """Tests for BenchmarkResults."""

    def test_to_dict(self, sample_run_records):
        """Test serialization to dictionary."""
        summaries = [
            ConfigSummary(
                config_id="baseline",
                config_name="Baseline",
                total_runs=2,
                success_count=1,
                success_rate=0.5,
                duration_mean=35.0,
                duration_p50=35.0,
                duration_p90=55.0,
                duration_std=25.0,
                tokens_mean=2250,
                tool_calls_mean=0.0,
                cost_mean=0.0165,
                cost_total=0.033,
            )
        ]

        results = BenchmarkResults(
            experiment_name="test",
            timestamp=datetime.now(),
            total_duration_sec=120.0,
            records=sample_run_records,
            summaries=summaries,
        )

        data = results.to_dict()

        assert data["experiment_name"] == "test"
        assert data["summary"]["total_runs"] == 3
        assert data["summary"]["successful_runs"] == 2
        assert len(data["configs"]) == 1
        assert len(data["records"]) == 3
