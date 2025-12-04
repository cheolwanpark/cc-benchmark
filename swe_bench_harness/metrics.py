"""Metrics collection and aggregation for benchmark results.

This module defines data structures for recording run results
and provides statistical aggregation across configurations.
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from swe_bench_harness.config import PluginConfig, PricingConfig


class FailureType(Enum):
    """Types of failures that can occur during benchmark execution."""

    NONE = "none"  # Success - no failure
    TIMEOUT = "timeout"  # Agent exceeded time limit
    API_ERROR = "api_error"  # Anthropic API error
    PARSE_ERROR = "parse_error"  # Failed to parse agent response
    AGENT_ERROR = "agent_error"  # Agent reported failure
    UNKNOWN = "unknown"  # Unexpected error


@dataclass
class RunRecord:
    """Record of a single benchmark run.

    Captures all metrics from executing an agent on one instance.
    """

    run_id: str
    instance_id: str
    config_id: str
    timestamp: datetime
    success: bool
    failure_type: FailureType = FailureType.NONE
    duration_sec: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0
    tool_calls_total: int = 0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    error_reason: str | None = None
    cost_usd: float = 0.0
    patch_generated: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "config_id": self.config_id,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "failure_type": self.failure_type.value,
            "duration_sec": self.duration_sec,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_cache_read": self.tokens_cache_read,
            "tool_calls_total": self.tool_calls_total,
            "tool_calls_by_name": self.tool_calls_by_name,
            "error_reason": self.error_reason,
            "cost_usd": self.cost_usd,
            "patch_generated": self.patch_generated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        """Create from dictionary (for deserialization)."""
        return cls(
            run_id=data["run_id"],
            instance_id=data["instance_id"],
            config_id=data["config_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            success=data["success"],
            failure_type=FailureType(data.get("failure_type", "none")),
            duration_sec=data.get("duration_sec", 0.0),
            tokens_input=data.get("tokens_input", 0),
            tokens_output=data.get("tokens_output", 0),
            tokens_cache_read=data.get("tokens_cache_read", 0),
            tool_calls_total=data.get("tool_calls_total", 0),
            tool_calls_by_name=data.get("tool_calls_by_name", {}),
            error_reason=data.get("error_reason"),
            cost_usd=data.get("cost_usd", 0.0),
            patch_generated=data.get("patch_generated"),
        )


@dataclass
class ConfigSummary:
    """Aggregated statistics for a single plugin configuration."""

    config_id: str
    config_name: str
    total_runs: int
    success_count: int
    success_rate: float
    duration_mean: float
    duration_p50: float
    duration_p90: float
    duration_std: float
    tokens_mean: int
    tool_calls_mean: float
    cost_mean: float
    cost_total: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_id": self.config_id,
            "config_name": self.config_name,
            "total_runs": self.total_runs,
            "success_count": self.success_count,
            "success_rate": round(self.success_rate, 4),
            "duration_mean": round(self.duration_mean, 2),
            "duration_p50": round(self.duration_p50, 2),
            "duration_p90": round(self.duration_p90, 2),
            "duration_std": round(self.duration_std, 2),
            "tokens_mean": self.tokens_mean,
            "tool_calls_mean": round(self.tool_calls_mean, 2),
            "cost_mean": round(self.cost_mean, 4),
            "cost_total": round(self.cost_total, 4),
        }


@dataclass
class BenchmarkResults:
    """Complete results from a benchmark experiment."""

    experiment_name: str
    timestamp: datetime
    total_duration_sec: float
    records: list[RunRecord]
    summaries: list[ConfigSummary]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp.isoformat(),
            "total_duration_sec": round(self.total_duration_sec, 2),
            "summary": {
                "total_runs": len(self.records),
                "successful_runs": sum(1 for r in self.records if r.success),
                "failed_runs": sum(1 for r in self.records if not r.success),
                "total_cost_usd": round(sum(r.cost_usd for r in self.records), 4),
            },
            "configs": [s.to_dict() for s in self.summaries],
            "records": [r.to_dict() for r in self.records],
        }


class MetricsAggregator:
    """Aggregates run records into configuration summaries."""

    def __init__(self, pricing: PricingConfig) -> None:
        """Initialize with pricing configuration.

        Args:
            pricing: Token pricing configuration
        """
        self.pricing = pricing

    def calculate_cost(
        self,
        tokens_input: int,
        tokens_output: int,
        tokens_cache_read: int = 0,
    ) -> float:
        """Calculate cost for a single run.

        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            tokens_cache_read: Number of cache read tokens

        Returns:
            Cost in USD
        """
        input_cost = (tokens_input / 1_000_000) * self.pricing.input_cost_per_mtok
        output_cost = (tokens_output / 1_000_000) * self.pricing.output_cost_per_mtok
        cache_cost = (tokens_cache_read / 1_000_000) * self.pricing.cache_read_cost_per_mtok
        return input_cost + output_cost + cache_cost

    def aggregate(
        self,
        records: list[RunRecord],
        configs: list[PluginConfig],
    ) -> list[ConfigSummary]:
        """Aggregate records by configuration.

        Args:
            records: List of all run records
            configs: List of plugin configurations

        Returns:
            List of ConfigSummary for each configuration
        """
        # Build config name lookup
        config_names = {c.id: c.name for c in configs}

        # Group records by config_id
        records_by_config: dict[str, list[RunRecord]] = {}
        for record in records:
            if record.config_id not in records_by_config:
                records_by_config[record.config_id] = []
            records_by_config[record.config_id].append(record)

        summaries = []
        for config_id, config_records in records_by_config.items():
            summary = self._summarize_config(
                config_id=config_id,
                config_name=config_names.get(config_id, config_id),
                records=config_records,
            )
            summaries.append(summary)

        # Sort by config_id for consistent ordering
        summaries.sort(key=lambda s: s.config_id)
        return summaries

    def _summarize_config(
        self,
        config_id: str,
        config_name: str,
        records: list[RunRecord],
    ) -> ConfigSummary:
        """Create summary statistics for a single configuration.

        Args:
            config_id: Configuration identifier
            config_name: Human-readable configuration name
            records: All run records for this configuration

        Returns:
            Aggregated ConfigSummary
        """
        if not records:
            return ConfigSummary(
                config_id=config_id,
                config_name=config_name,
                total_runs=0,
                success_count=0,
                success_rate=0.0,
                duration_mean=0.0,
                duration_p50=0.0,
                duration_p90=0.0,
                duration_std=0.0,
                tokens_mean=0,
                tool_calls_mean=0.0,
                cost_mean=0.0,
                cost_total=0.0,
            )

        total_runs = len(records)
        success_count = sum(1 for r in records if r.success)
        success_rate = success_count / total_runs if total_runs > 0 else 0.0

        # Duration statistics
        durations = [r.duration_sec for r in records]
        duration_mean = statistics.mean(durations)
        duration_p50 = self._percentile(durations, 50)
        duration_p90 = self._percentile(durations, 90)
        duration_std = statistics.stdev(durations) if len(durations) > 1 else 0.0

        # Token statistics (total = input + output)
        total_tokens = [r.tokens_input + r.tokens_output for r in records]
        tokens_mean = int(statistics.mean(total_tokens))

        # Tool call statistics
        tool_calls = [r.tool_calls_total for r in records]
        tool_calls_mean = statistics.mean(tool_calls)

        # Cost statistics
        costs = [r.cost_usd for r in records]
        cost_mean = statistics.mean(costs)
        cost_total = sum(costs)

        return ConfigSummary(
            config_id=config_id,
            config_name=config_name,
            total_runs=total_runs,
            success_count=success_count,
            success_rate=success_rate,
            duration_mean=duration_mean,
            duration_p50=duration_p50,
            duration_p90=duration_p90,
            duration_std=duration_std,
            tokens_mean=tokens_mean,
            tool_calls_mean=tool_calls_mean,
            cost_mean=cost_mean,
            cost_total=cost_total,
        )

    @staticmethod
    def _percentile(data: list[float], percentile: int) -> float:
        """Calculate percentile from a list of values.

        Args:
            data: List of numeric values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not data:
            return 0.0

        sorted_data = sorted(data)
        n = len(sorted_data)

        if n == 1:
            return sorted_data[0]

        # Calculate index
        index = (percentile / 100) * (n - 1)
        lower = int(index)
        upper = lower + 1
        weight = index - lower

        if upper >= n:
            return sorted_data[-1]

        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
