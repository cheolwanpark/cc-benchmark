"""TUI package for cc-benchmark.

This package provides a Rich-based terminal user interface for running
SWE-bench benchmarks interactively or with configuration files.
"""

from cc_benchmark.tui.app import main, run_tui
from cc_benchmark.tui.progress import BenchmarkProgress, BenchmarkStats
from cc_benchmark.tui.wizard import ConfigWizard

__all__ = [
    "main",
    "run_tui",
    "BenchmarkProgress",
    "BenchmarkStats",
    "ConfigWizard",
]
