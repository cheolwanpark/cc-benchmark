#!/usr/bin/env python3
"""SWE-Bench Plugin Efficiency Benchmark Tool.

This is a convenience entry point. For full CLI options, use:
    cc-bench-run --help

Or run directly:
    python -m scripts.benchmark --config examples/experiment.yaml
"""

import sys

from cc_benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main())
