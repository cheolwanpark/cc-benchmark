"""Command-line interface for the benchmark harness."""

import argparse
import asyncio
import sys
from datetime import datetime

from cc_benchmark.config import Config
from cc_benchmark.dataset import load_instances
from cc_benchmark.metrics import BenchmarkResults
from cc_benchmark.reporter import save_results
from cc_benchmark.runner import run_benchmark


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SWE-Bench Benchmark Tool",
    )
    parser.add_argument(
        "config",
        help="Path to YAML experiment configuration file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Override output directory from config",
    )

    args = parser.parse_args()

    try:
        return asyncio.run(run_cli(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


async def run_cli(args: argparse.Namespace) -> int:
    """Run the CLI with parsed arguments."""
    # Load configuration
    print(f"Loading configuration from: {args.config}")

    try:
        config = Config.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir

    # Load dataset
    print(f"Loading dataset: {config.dataset.source}")
    try:
        instances = load_instances(config.dataset)
        print(f"Loaded {len(instances)} instances")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Run benchmark
    print(f"\nRunning benchmark with {config.execution.max_parallel} parallel workers...")
    start_time = datetime.now()
    records = []

    def on_progress(completed: int, total: int, instance_id: str) -> None:
        print(f"[{completed}/{total}] Completed: {instance_id}")

    async for record in run_benchmark(config, instances, on_progress=on_progress, verbose=False):
        records.append(record)

    total_duration = (datetime.now() - start_time).total_seconds()

    # Build results
    results = BenchmarkResults(
        experiment_name=config.name,
        timestamp=start_time,
        total_duration_sec=total_duration,
        records=records,
    )

    # Save results
    output_path = config.get_output_path()
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = save_results(results, output_path)
    print(f"\nResults saved to: {report_path}")

    # Print summary
    total_runs = len(records)
    patches = sum(1 for r in records if r.patch)
    resolved = sum(1 for r in records if r.resolved)
    total_cost = sum(r.cost_usd for r in records)

    print(f"\nSummary:")
    print(f"  Total runs: {total_runs}")
    print(f"  Patches generated: {patches}/{total_runs} ({100*patches/total_runs:.1f}%)")
    print(f"  Resolved: {resolved}/{total_runs} ({100*resolved/total_runs:.1f}%)")
    print(f"  Duration: {total_duration/60:.1f} minutes")
    print(f"  Total cost: ${total_cost:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
