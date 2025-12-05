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
    BenchmarkConfig,
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelConfig,
    Plugin,
)
from swe_bench_harness.dataset import DatasetLoader, SWEBenchInstance
from swe_bench_harness.metrics import (
    BenchmarkResults,
    ConfigSummary,
    FailureType,
    MetricsAggregator,
    RunRecord,
)
from swe_bench_harness.agent import ClaudeAgent, ExecutionResult, SDK_TOOLS
from swe_bench_harness.evaluation import Evaluation, EvaluationError, EvaluationResult
from swe_bench_harness.images import (
    ImageError,
    ImageInfo,
    ImageNotFoundError,
    ImagePullError,
    Images,
)
from swe_bench_harness.runner import BenchmarkRunner, ProgressEvent
from swe_bench_harness.reporter import Reporter
from swe_bench_harness.validator import ConfigValidator
from swe_bench_harness.plugins import PluginLoader, plugin_context

__all__ = [
    # Version
    "__version__",
    # Config
    "BenchmarkConfig",
    "DatasetConfig",
    "ExecutionConfig",
    "ExperimentConfig",
    "ModelConfig",
    "Plugin",
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
    "SDK_TOOLS",
    # Evaluation
    "Evaluation",
    "EvaluationError",
    "EvaluationResult",
    # Images
    "ImageError",
    "ImageInfo",
    "ImageNotFoundError",
    "ImagePullError",
    "Images",
    # Runner
    "BenchmarkRunner",
    "ProgressEvent",
    # Reporter
    "Reporter",
    # Validator
    "ConfigValidator",
    # Plugins
    "PluginLoader",
    "plugin_context",
]
