"""Metrics collection for benchmark results."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class FailureType(Enum):
    """Types of failures that can occur during benchmark execution."""

    NONE = "none"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    AGENT_ERROR = "agent_error"
    UNKNOWN = "unknown"


@dataclass
class RunRecord:
    """Record of a single benchmark run."""

    instance_id: str
    timestamp: datetime
    success: bool
    failure_type: FailureType = FailureType.NONE
    duration_sec: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0
    tool_calls_total: int = 0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    error: str | None = None
    patch: str | None = None
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "instance_id": self.instance_id,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "failure_type": self.failure_type.value,
            "duration_sec": round(self.duration_sec, 2),
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_cache_read": self.tokens_cache_read,
            "tool_calls_total": self.tool_calls_total,
            "tool_calls_by_name": self.tool_calls_by_name,
            "cost_usd": round(self.cost_usd, 6),
            "error": self.error,
            "patch": self.patch,
            "resolved": self.resolved,
        }


@dataclass
class BenchmarkResults:
    """Complete results from a benchmark experiment."""

    experiment_name: str
    timestamp: datetime
    total_duration_sec: float
    records: list[RunRecord]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        total_runs = len(self.records)
        patches = sum(1 for r in self.records if r.patch)
        resolved = sum(1 for r in self.records if r.resolved)
        total_cost = sum(r.cost_usd for r in self.records)

        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp.isoformat(),
            "total_duration_sec": round(self.total_duration_sec, 2),
            "summary": {
                "total_runs": total_runs,
                "patches_generated": patches,
                "patch_rate": round(patches / total_runs, 4) if total_runs > 0 else 0,
                "resolved": resolved,
                "resolve_rate": round(resolved / total_runs, 4) if total_runs > 0 else 0,
                "total_cost_usd": round(total_cost, 4),
            },
            "records": [r.to_dict() for r in self.records],
        }
