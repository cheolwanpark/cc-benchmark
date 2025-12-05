"""SWE-Bench Plugin Efficiency Benchmark Tool.

A benchmarking harness that measures plugin/tool efficiency on SWE-bench across:
- Success Rate (solve rate percentage)
- Execution Time (seconds per task)
- Tool Call Efficiency (API calls per task)
- Token Usage (input + output tokens, cost projection)
"""

__version__ = "1.0.0"

# Imports will be added as modules are implemented
from cc_benchmark.config import (
    BenchmarkConfig,
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelConfig,
    Plugin,
)
from cc_benchmark.dataset import DatasetLoader, SWEBenchInstance
from cc_benchmark.metrics import (
    BenchmarkResults,
    ConfigSummary,
    FailureType,
    MetricsAggregator,
    RunRecord,
)
from cc_benchmark.agent import DockerClaudeAgent, ExecutionResult, SDK_TOOLS
from cc_benchmark.evaluation import Evaluation, EvaluationError, EvaluationResult
from cc_benchmark.images import (
    ImageError,
    ImageInfo,
    ImageNotFoundError,
    ImagePullError,
    Images,
)
from cc_benchmark.runner import BenchmarkRunner, ProgressEvent
from cc_benchmark.reporter import Reporter
from cc_benchmark.validator import ConfigValidator
from cc_benchmark.plugins import PluginLoader, plugin_context

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
    "DockerClaudeAgent",
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
