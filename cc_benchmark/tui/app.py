"""Main TUI application orchestrator for cc-benchmark.

This module provides the main entry point and orchestration logic
for the TUI application, handling both config-file mode and
interactive wizard mode.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cc_benchmark.config import Config, DatasetConfig
from cc_benchmark.dataset import load_instances
from cc_benchmark.metrics import BenchmarkResults
from cc_benchmark.reporter import save_results
from cc_benchmark.runner import run_benchmark
from cc_benchmark.test_run import run_test_instance
from cc_benchmark.tui.progress import BenchmarkProgress
from cc_benchmark.tui.styles import STYLES, format_cost, format_rate
from cc_benchmark.tui.wizard import ConfigWizard

# Initialize console with theme
console = Console()


def main() -> int:
    """CLI entry point with mode detection.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        prog="cc-bench",
        description="SWE-Bench Benchmark Tool with interactive wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cc-bench                      # Interactive wizard mode
  cc-bench config.yaml          # Run with config file
  cc-bench config.yaml -o out/  # Override output directory
""",
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to YAML config file (omit for interactive wizard)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Override output directory from config",
    )

    args = parser.parse_args()

    # Check for required OAuth token
    if not _check_oauth_token():
        return 1

    try:
        return asyncio.run(run_tui(args.config, args.output_dir))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130


def _check_oauth_token() -> bool:
    """Check if CLAUDE_CODE_OAUTH_TOKEN environment variable is set.

    Returns:
        True if token is set, False otherwise
    """
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if not token:
        console.print(Panel(
            "[red]Error:[/red] CLAUDE_CODE_OAUTH_TOKEN environment variable is not set.\n\n"
            "This token is required to authenticate with the Claude API.\n"
            "Please set it before running cc-bench:\n\n"
            "  [cyan]export CLAUDE_CODE_OAUTH_TOKEN=your_token_here[/cyan]",
            title="Missing OAuth Token",
            style=STYLES["error_panel"],
        ))
        return False
    return True


async def run_tui(config_path: str | None, output_dir_override: str | None) -> int:
    """Main TUI execution flow.

    Args:
        config_path: Path to config file, or None for interactive wizard
        output_dir_override: Optional override for output directory

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Display welcome banner
    _show_welcome()

    if config_path:
        # Mode 1: Config file provided
        config = await _load_config_mode(config_path, output_dir_override)
    else:
        # Mode 2: Interactive wizard
        config = await _wizard_mode()

    if config is None:
        return 1

    # Execute benchmark with progress display
    return await _run_benchmark_with_progress(config)


def _show_welcome() -> None:
    """Display welcome panel."""
    console.print(Panel(
        "[bold blue]cc-benchmark[/bold blue]\n"
        "[dim]SWE-Bench Plugin Efficiency Benchmark Tool[/dim]",
        style=STYLES["welcome"],
        padding=(1, 2),
    ))


async def _load_config_mode(
    path: str, output_override: str | None
) -> Config | None:
    """Load and validate config from file.

    Args:
        path: Path to config file
        output_override: Optional output directory override

    Returns:
        Config object or None on error
    """
    console.print(f"Loading configuration from: [cyan]{path}[/cyan]")

    try:
        config = Config.from_yaml(path)
        if output_override:
            config.output_dir = output_override
        _display_config_summary(config)
        return config

    except FileNotFoundError:
        console.print(Panel(
            f"[red]Error:[/red] Config file not found: {path}",
            title="Error",
            style=STYLES["error_panel"],
        ))
        return None

    except Exception as e:
        console.print(Panel(
            f"[red]Error:[/red] Failed to load config: {e}",
            title="Error",
            style=STYLES["error_panel"],
        ))
        return None


async def _wizard_mode() -> Config | None:
    """Run interactive wizard to create config.

    Returns:
        Config object or None if cancelled
    """
    wizard = ConfigWizard(console)
    config = wizard.run()  # Synchronous - Rich prompts are blocking

    if config is None:
        return None

    # Ask to save config
    if wizard.prompt_save_config():
        filename = wizard.prompt_filename()
        try:
            _save_config_yaml(config, filename)
            console.print(f"[green]Config saved to:[/green] {filename}")
        except Exception as e:
            console.print(f"[red]Failed to save config:[/red] {e}")
            # Don't fail the whole operation

    # Ask for test run
    if wizard.prompt_test_run():
        console.print("\n[yellow]Running single-instance test first...[/yellow]")

        # Create test config with single instance
        test_config = _create_test_config(config)

        # Load first instance
        try:
            instances = load_instances(test_config.dataset)
            if not instances:
                console.print("[red]No instances found in dataset[/red]")
                return config

            instance = instances[0]

            # Run test
            test_result = await run_test_instance(
                instance=instance,
                config=test_config,
                console=console,
                skip_evaluation=False,
            )

            # Check result
            if not test_result.success:
                if not wizard.prompt_continue_after_test_failure():
                    console.print("[yellow]Benchmark cancelled after test failure.[/yellow]")
                    return None

        except Exception as e:
            console.print(f"[red]Test run error:[/red] {e}")
            if not wizard.prompt_continue_after_test_failure():
                return None

    return config


def _create_test_config(config: Config) -> Config:
    """Create config for single-instance test run.

    Args:
        config: Original config

    Returns:
        Config modified for single instance
    """
    return Config(
        name=f"{config.name}-test",
        dataset=DatasetConfig(
            name=config.dataset.name,
            split=":1",  # Only first instance
            cache_dir=config.dataset.cache_dir,
        ),
        execution=config.execution,
        model=config.model,
        plugins=config.plugins,
        output_dir=config.output_dir,
    )


def _display_config_summary(config: Config) -> None:
    """Display config in Rich table.

    Args:
        config: Config to display
    """
    table = Table(title="Configuration Summary", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Experiment Name", config.name)
    table.add_row("Dataset", f"{config.dataset.name} ({config.dataset.source})")
    table.add_row("Split", config.dataset.split)
    table.add_row("Model", config.model)
    table.add_row("Parallelism", str(config.execution.max_parallel))
    table.add_row("Timeout", f"{config.execution.timeout_sec}s")
    table.add_row("Output Directory", config.output_dir)

    if config.plugins:
        table.add_row("Plugins", ", ".join(config.plugins))

    console.print(table)


def _save_config_yaml(config: Config, filename: str) -> None:
    """Save config to YAML file.

    Args:
        config: Config to save
        filename: Output filename
    """
    data = {
        "name": config.name,
        "dataset": {
            "name": config.dataset.name,
            "split": config.dataset.split,
        },
        "execution": {
            "max_parallel": config.execution.max_parallel,
            "timeout_sec": config.execution.timeout_sec,
        },
        "model": config.model,
        "output_dir": config.output_dir,
    }

    # Add optional fields if non-default
    if config.execution.docker_memory != "8g":
        data["execution"]["docker_memory"] = config.execution.docker_memory
    if config.execution.docker_cpus != 4:
        data["execution"]["docker_cpus"] = config.execution.docker_cpus
    if config.plugins:
        data["plugins"] = config.plugins

    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


async def _run_benchmark_with_progress(config: Config) -> int:
    """Execute benchmark with Rich progress display.

    Args:
        config: Benchmark configuration

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Load dataset
    console.print(f"\nLoading dataset: [cyan]{config.dataset.source}[/cyan]")

    try:
        instances = load_instances(config.dataset)
        console.print(f"Loaded [green]{len(instances)}[/green] instances")
    except Exception as e:
        console.print(Panel(
            f"[red]Error loading dataset:[/red] {e}",
            title="Error",
            style=STYLES["error_panel"],
        ))
        return 1

    if not instances:
        console.print("[yellow]No instances to process[/yellow]")
        return 0

    workers = config.execution.max_parallel
    console.print(f"\nRunning benchmark with [cyan]{workers}[/cyan] parallel workers...")
    console.print()

    # Create progress display
    progress_display = BenchmarkProgress(console, len(instances))

    # Run benchmark
    start_time = datetime.now()
    records = []

    try:
        with progress_display:
            async for record in run_benchmark(
                config,
                instances,
                on_progress=progress_display.on_progress,
                verbose=False,
            ):
                records.append(record)
                progress_display.update(record)

    except KeyboardInterrupt:
        progress_display.stop()
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        # Still save partial results
        if records:
            console.print("[dim]Saving partial results...[/dim]")

    except Exception as e:
        progress_display.stop()
        console.print(Panel(
            f"[red]Benchmark error:[/red] {e}",
            title="Error",
            style=STYLES["error_panel"],
        ))
        # Still try to save partial results
        if not records:
            return 1

    total_duration = (datetime.now() - start_time).total_seconds()

    # Build and save results
    results = BenchmarkResults(
        experiment_name=config.name,
        timestamp=start_time,
        total_duration_sec=total_duration,
        records=records,
    )

    output_path = config.get_output_path()
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        report_path = save_results(results, output_path)
    except Exception as e:
        console.print(f"[red]Failed to save results:[/red] {e}")
        report_path = None

    # Display final summary
    _display_final_summary(results, report_path)

    return 0


def _display_final_summary(
    results: BenchmarkResults, report_path: Path | None
) -> None:
    """Display final benchmark summary.

    Args:
        results: Benchmark results
        report_path: Path to saved results file
    """
    total = len(results.records)
    if total == 0:
        console.print("[yellow]No results to summarize[/yellow]")
        return

    patches = sum(1 for r in results.records if r.patch)
    resolved = sum(1 for r in results.records if r.resolved)
    failed = sum(1 for r in results.records if not r.success)
    total_cost = sum(r.cost_usd for r in results.records)

    # Build summary table
    summary = Table.grid(padding=1)
    summary.add_column(justify="right", style="cyan")
    summary.add_column(justify="left")

    summary.add_row("Total runs:", str(total))
    summary.add_row(
        "Patches generated:",
        f"{patches}/{total} ({format_rate(patches / total)})",
    )
    summary.add_row(
        "Resolved:",
        f"{resolved}/{total} ({format_rate(resolved / total)})",
    )
    if failed > 0:
        summary.add_row("Failed:", f"[red]{failed}[/red]")
    summary.add_row(
        "Duration:",
        f"{results.total_duration_sec / 60:.1f} minutes",
    )
    summary.add_row("Total cost:", format_cost(total_cost))

    if report_path:
        summary.add_row("Results saved to:", str(report_path))

    console.print()
    console.print(Panel(
        summary,
        title="[bold green]Benchmark Complete[/bold green]",
        style=STYLES["success_panel"],
    ))


if __name__ == "__main__":
    sys.exit(main())
