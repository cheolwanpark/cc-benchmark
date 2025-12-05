"""SWE-Bench Benchmark Tool.

A simplified benchmarking harness for running Claude agents on SWE-bench.
"""

__version__ = "2.0.0"

from cc_benchmark.config import Config, DatasetConfig, ExecutionConfig
from cc_benchmark.dataset import SWEBenchInstance, load_instances
from cc_benchmark.metrics import BenchmarkResults, FailureType, RunRecord
from cc_benchmark.agent import AgentResult, run_agent
from cc_benchmark.evaluation import EvaluationResult, evaluate
from cc_benchmark.runner import run_benchmark
from cc_benchmark.reporter import save_results
from cc_benchmark.plugins import resolve_plugin_paths

__all__ = [
    "__version__",
    # Config
    "Config",
    "DatasetConfig",
    "ExecutionConfig",
    # Dataset
    "SWEBenchInstance",
    "load_instances",
    # Metrics
    "BenchmarkResults",
    "FailureType",
    "RunRecord",
    # Agent
    "AgentResult",
    "run_agent",
    # Evaluation
    "EvaluationResult",
    "evaluate",
    # Runner
    "run_benchmark",
    # Reporter
    "save_results",
    # Plugins
    "resolve_plugin_paths",
]
