#!/usr/bin/env python3
"""cc-benchmark TUI - Unified benchmark runner with interactive mode.

This is the main entry point for the cc-benchmark tool, combining
the functionality of the former benchmark.py and single_run.py scripts.

Usage:
    uv run cc-bench                  # Interactive wizard mode
    uv run cc-bench config.yaml      # Run with config file
    uv run cc-bench config.yaml -o out/  # Override output directory
    uv run cc-bench --help           # Show help
"""

import sys

from cc_benchmark.tui.app import main

if __name__ == "__main__":
    sys.exit(main())
