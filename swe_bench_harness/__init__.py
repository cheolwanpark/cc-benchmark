"""SWE-Bench Plugin Efficiency Benchmark Tool.

A benchmarking harness that measures plugin/tool efficiency on SWE-bench across:
- Success Rate (solve rate percentage)
- Execution Time (seconds per task)
- Tool Call Efficiency (API calls per task)
- Token Usage (input + output tokens, cost projection)
"""

__version__ = "1.0.0"

# Imports will be added as modules are implemented
from swe_bench_harness.config import (
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelConfig,
    PluginConfig,
    PricingConfig,
)
from swe_bench_harness.dataset import DatasetLoader, SWEBenchInstance
from swe_bench_harness.metrics import (
    BenchmarkResults,
    ConfigSummary,
    FailureType,
    MetricsAggregator,
    RunRecord,
)
from swe_bench_harness.agent import ClaudeAgent, ExecutionResult
from swe_bench_harness.runner import BenchmarkRunner, ProgressEvent
from swe_bench_harness.reporter import Reporter
from swe_bench_harness.validator import ConfigValidator

__all__ = [
    # Version
    "__version__",
    # Config
    "DatasetConfig",
    "ExecutionConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PluginConfig",
    "PricingConfig",
    # Dataset
    "DatasetLoader",
    "SWEBenchInstance",
    # Metrics
    "BenchmarkResults",
    "ConfigSummary",
    "FailureType",
    "MetricsAggregator",
    "RunRecord",
    # Agent
    "ClaudeAgent",
    "ExecutionResult",
    # Runner
    "BenchmarkRunner",
    "ProgressEvent",
    # Reporter
    "Reporter",
    # Validator
    "ConfigValidator",
]
