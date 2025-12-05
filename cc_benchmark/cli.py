"""Command-line interface for the benchmark harness.

This module provides the CLI entry point with Rich-based progress display.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from cc_benchmark.config import ExperimentConfig
from cc_benchmark.dataset import DatasetLoader
from cc_benchmark.metrics import BenchmarkResults
from cc_benchmark.reporter import Reporter
from cc_benchmark.runner import BenchmarkRunner, ProgressEvent
from cc_benchmark.validator import ConfigValidator

console = Console()


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="SWE-Bench Plugin Efficiency Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cc-bench-run --config experiment.yaml
  cc-bench-run --config experiment.yaml --dry-run
  cc-bench-run --config experiment.yaml --output-dir ./my-results
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML experiment configuration file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Override output directory from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running benchmark",
    )
    parser.add_argument(
        "--resume",
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    try:
        return run_cli(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if args.verbose:
            console.print_exception()
        return 1


def run_cli(args: argparse.Namespace) -> int:
    """Run the CLI with parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    # Load configuration
    console.print(f"[blue]Loading configuration from:[/blue] {args.config}")

    try:
        config = ExperimentConfig.from_yaml(args.config)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Configuration file not found: {args.config}")
        return 1
    except Exception as e:
        console.print(f"[red]Error loading configuration:[/red] {e}")
        return 1

    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir

    # Run pre-flight validation
    console.print("\n[blue]Running pre-flight checks...[/blue]")
    validator = ConfigValidator()
    errors = validator.run_all_checks(config)
    warnings = validator.get_warnings()

    if errors:
        console.print("[red]Pre-flight checks failed:[/red]")
        for error in errors:
            console.print(f"  [red]•[/red] {error}")
        return 1

    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  [yellow]•[/yellow] {warning}")

    # Display validation summary
    summary = validator.get_summary(config)
    print_validation_summary(summary)

    if args.dry_run:
        console.print("\n[green]✓ Validation passed (dry run)[/green]")
        return 0

    # Load dataset
    console.print("\n[blue]Loading dataset...[/blue]")
    loader = DatasetLoader()
    try:
        instances = loader.load(config.dataset)
        console.print(f"[green]✓ Loaded {len(instances)} instances[/green]")
    except Exception as e:
        console.print(f"[red]Error loading dataset:[/red] {e}")
        return 1

    # Use resolved output path
    output_path = config.get_output_path()

    # Initialize runner
    checkpoint_dir = output_path / "checkpoints" if args.resume else None
    runner = BenchmarkRunner(config, instances, checkpoint_dir=checkpoint_dir)

    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            loaded = runner.load_checkpoint(resume_path)
            console.print(f"[green]✓ Resumed from checkpoint ({loaded} runs)[/green]")

    # Run benchmark with progress
    console.print("\n[blue]Running benchmark...[/blue]\n")

    results = asyncio.run(run_with_progress(runner))

    # Generate reports
    console.print("\n[blue]Generating reports...[/blue]")
    reporter = Reporter(output_path)
    paths = reporter.generate_all(results)

    console.print("[green]✓ Reports generated:[/green]")
    for format_name, path in paths.items():
        console.print(f"  • {format_name}: {path}")

    # Print summary table
    console.print()
    print_summary_table(results)

    return 0


async def run_with_progress(runner: BenchmarkRunner) -> BenchmarkResults:
    """Run benchmark with Rich progress display.

    Args:
        runner: Configured benchmark runner

    Returns:
        Benchmark results
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )

    with progress:
        task_id = progress.add_task(
            "[cyan]Benchmarking...",
            total=runner.total_runs,
        )

        def on_progress(event: ProgressEvent) -> None:
            progress.update(
                task_id,
                completed=event.completed,
                description=f"[cyan]{event.current_config}[/cyan] - {event.current_instance}",
            )

        async for _ in runner.run(on_progress=on_progress):
            pass

    return runner.get_results()


def print_validation_summary(summary: dict) -> None:
    """Print validation summary as a panel.

    Args:
        summary: Validation summary dictionary
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")

    table.add_row("Dataset", summary["dataset_source"])
    table.add_row("Instances", str(summary["estimated_instances"]))
    table.add_row("Configurations", str(summary["num_configs"]))
    table.add_row("Runs per instance", str(summary["runs"]))
    table.add_row("Total runs", str(summary["total_runs"]))
    table.add_row("Parallel tasks", str(summary["max_parallel"]))
    table.add_row("Timeout per run", f"{summary['timeout_sec']}s")
    table.add_row("Estimated cost", f"${summary['estimated_cost_usd']:.2f}")
    table.add_row("Output directory", summary["output_dir"])

    panel = Panel(table, title="[bold]Experiment Configuration[/bold]", border_style="blue")
    console.print(panel)


def print_summary_table(results: BenchmarkResults) -> None:
    """Print final results as a Rich table.

    Args:
        results: Benchmark results
    """
    table = Table(title=f"[bold]{results.experiment_name}[/bold] - Results Summary")

    table.add_column("Configuration", style="cyan")
    table.add_column("Patch Rate", justify="right")
    table.add_column("Resolve Rate", justify="right")
    table.add_column("Avg Duration", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Avg Tool Calls", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Total Cost", justify="right")

    for summary in results.summaries:
        # Color code patch rate
        patch_rate = summary.patch_rate * 100
        if patch_rate >= 70:
            patch_style = "green"
        elif patch_rate >= 40:
            patch_style = "yellow"
        else:
            patch_style = "red"

        # Color code resolve rate
        resolve_rate = summary.resolve_rate * 100
        if resolve_rate >= 50:
            resolve_style = "green"
        elif resolve_rate >= 20:
            resolve_style = "yellow"
        else:
            resolve_style = "red"

        table.add_row(
            summary.config_name,
            f"[{patch_style}]{patch_rate:.1f}%[/{patch_style}]",
            f"[{resolve_style}]{resolve_rate:.1f}%[/{resolve_style}]",
            f"{summary.duration_mean:.1f}s",
            f"{summary.tokens_mean:,}",
            f"{summary.tool_calls_mean:.1f}",
            f"${summary.cost_mean:.4f}",
            f"${summary.cost_total:.2f}",
        )

    console.print(table)

    # Print totals
    total_runs = len(results.records)
    patches_generated = sum(1 for r in results.records if r.patch_generated)
    resolved = sum(1 for r in results.records if r.resolved)
    total_cost = sum(r.cost_usd for r in results.records)

    console.print()
    console.print(f"[bold]Total:[/bold] {patches_generated}/{total_runs} patches generated, {resolved}/{total_runs} resolved")
    console.print(f"[bold]Duration:[/bold] {results.total_duration_sec / 60:.1f} minutes")
    console.print(f"[bold]Total Cost:[/bold] ${total_cost:.2f}")


if __name__ == "__main__":
    sys.exit(main())
