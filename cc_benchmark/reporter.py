"""Report generation for benchmark results."""

import json
from pathlib import Path

from cc_benchmark.metrics import BenchmarkResults


def save_results(results: BenchmarkResults, output_dir: Path) -> Path:
    """Save benchmark results to JSON file.

    Args:
        results: Benchmark results to save
        output_dir: Directory for output file

    Returns:
        Path to generated file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"

    with open(path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    return path
