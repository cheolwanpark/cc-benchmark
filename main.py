#!/usr/bin/env python3
"""SWE-Bench Plugin Efficiency Benchmark Tool.

This is a convenience entry point. For full CLI options, use:
    swe-bench-harness --help

Or run directly:
    python main.py --config examples/experiment.yaml
"""

import sys

from swe_bench_harness.cli import main

if __name__ == "__main__":
    sys.exit(main())
