"""Single instance test run for debugging and validation.

This module provides a reusable function for running a single SWE-bench instance
with full visibility into the execution process. It uses Rich Console for output
instead of print() to integrate properly with TUI applications.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cc_benchmark.agent import AgentResult, run_agent
from cc_benchmark.config import Config
from cc_benchmark.dataset import SWEBenchInstance
from cc_benchmark.evaluation import EvaluationResult, evaluate
from cc_benchmark.metrics import FailureType


@dataclass
class TestRunResult:
    """Complete result of a single test run including all metrics."""

    instance_id: str
    success: bool
    failure_type: FailureType = FailureType.NONE

    # Agent execution metrics
    duration_sec: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0
    tool_calls_total: int = 0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0

    # Results
    patch: str | None = None
    resolved: bool | None = None  # None = not evaluated
    error: str | None = None

    # Evaluation metrics
    eval_duration_sec: float = 0.0
    eval_error: str | None = None

    @classmethod
    def from_agent_result(
        cls, instance_id: str, agent_result: AgentResult
    ) -> TestRunResult:
        """Create TestRunResult from an AgentResult."""
        return cls(
            instance_id=instance_id,
            success=agent_result.success,
            failure_type=agent_result.failure_type,
            duration_sec=agent_result.duration_sec,
            tokens_input=agent_result.tokens_input,
            tokens_output=agent_result.tokens_output,
            tokens_cache_read=agent_result.tokens_cache_read,
            tool_calls_total=agent_result.tool_calls_total,
            tool_calls_by_name=agent_result.tool_calls_by_name,
            cost_usd=agent_result.cost_usd,
            patch=agent_result.patch,
            error=agent_result.error,
        )

    def update_from_evaluation(self, eval_result: EvaluationResult) -> None:
        """Update with evaluation results."""
        self.resolved = eval_result.resolved
        self.eval_duration_sec = eval_result.duration_sec
        self.eval_error = eval_result.error


async def run_test_instance(
    instance: SWEBenchInstance,
    config: Config,
    console: Console,
    skip_evaluation: bool = False,
) -> TestRunResult:
    """Run a single SWE-bench instance with full visibility.

    This function executes the complete test flow:
    1. Clone repository and checkout base commit
    2. Run agent in Docker container
    3. Optionally evaluate generated patch

    Args:
        instance: SWE-bench instance to test
        config: Benchmark configuration
        console: Rich Console for output (no global print())
        skip_evaluation: If True, skip the evaluation step

    Returns:
        TestRunResult with all execution metrics
    """
    console.print()
    console.print(Panel(
        f"[bold]Test Run: {instance.instance_id}[/bold]",
        style="cyan",
    ))

    # Display instance info
    _display_instance_info(console, instance)

    # Step 1: Setup workspace
    console.print("\n[bold cyan]Step 1/4:[/bold cyan] Setting up repository...")
    base_dir = Path(tempfile.mkdtemp(prefix="test_run_"))
    work_dir = base_dir / "repo"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        clone_success = await _clone_and_checkout(
            instance=instance,
            work_dir=work_dir,
            console=console,
        )

        if not clone_success:
            return TestRunResult(
                instance_id=instance.instance_id,
                success=False,
                failure_type=FailureType.UNKNOWN,
                error="Failed to clone repository or checkout commit",
            )

        # Step 2: Run agent
        console.print("\n[bold cyan]Step 2/4:[/bold cyan] Running agent in Docker...")
        console.print(f"  Docker image: [dim]{config.execution.docker_image}[/dim]")
        console.print(f"  Model: [dim]{config.model}[/dim]")
        console.print(f"  Timeout: [dim]{config.execution.timeout_sec}s[/dim]")
        console.print()

        agent_result = await run_agent(
            instance=instance,
            config=config,
            work_dir=work_dir,
            output_dir=output_dir,
            verbose=True,  # Show agent output for debugging
        )

        result = TestRunResult.from_agent_result(instance.instance_id, agent_result)

        # Step 3: Display agent results
        console.print("\n[bold cyan]Step 3/4:[/bold cyan] Agent execution complete")
        _display_agent_results(console, result)

        # Step 4: Evaluate patch (if generated and not skipped)
        if result.patch and not skip_evaluation:
            console.print("\n[bold cyan]Step 4/4:[/bold cyan] Running SWE-bench evaluation...")
            console.print("  [dim](This may take several minutes)[/dim]")

            eval_result = await evaluate(instance, result.patch, config)
            result.update_from_evaluation(eval_result)

            _display_evaluation_result(console, result)
        elif not result.patch:
            console.print(
                "\n[bold cyan]Step 4/4:[/bold cyan] [yellow]Skipped (no patch)[/yellow]"
            )
        else:
            console.print(
                "\n[bold cyan]Step 4/4:[/bold cyan] [yellow]Skipped (disabled)[/yellow]"
            )

        # Final summary
        _display_summary(console, result)

        return result

    finally:
        # Cleanup
        console.print("\n[dim]Cleaning up temporary files...[/dim]")
        shutil.rmtree(base_dir, ignore_errors=True)


def _display_instance_info(console: Console, instance: SWEBenchInstance) -> None:
    """Display instance information in a table."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="cyan")
    table.add_column("Value")

    table.add_row("Instance ID:", instance.instance_id)
    table.add_row("Repository:", instance.repo)
    table.add_row("Base Commit:", f"{instance.base_commit[:12]}...")

    console.print(table)

    # Show truncated problem statement
    console.print("\n[bold]Problem Statement:[/bold]")
    problem_preview = instance.problem_statement[:500]
    if len(instance.problem_statement) > 500:
        problem_preview += "..."
    console.print(Panel(problem_preview, style="dim"))


def _display_agent_results(console: Console, result: TestRunResult) -> None:
    """Display agent execution results."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    status_style = "green" if result.success else "red"
    table.add_row("Success:", f"[{status_style}]{result.success}[/{status_style}]")

    if result.failure_type != FailureType.NONE:
        table.add_row("Failure Type:", f"[yellow]{result.failure_type.name}[/yellow]")

    table.add_row("Duration:", f"{result.duration_sec:.2f}s")
    table.add_row("Input Tokens:", f"{result.tokens_input:,}")
    table.add_row("Output Tokens:", f"{result.tokens_output:,}")
    table.add_row("Cache Read:", f"{result.tokens_cache_read:,}")
    table.add_row("Tool Calls:", str(result.tool_calls_total))
    table.add_row("Cost:", f"${result.cost_usd:.4f}")

    console.print(table)

    if result.tool_calls_by_name:
        console.print("\n[bold]Tool breakdown:[/bold]")
        for name, count in sorted(result.tool_calls_by_name.items()):
            console.print(f"  {name}: {count}")

    if result.error:
        console.print(f"\n[red]Error:[/red] {result.error}")

    if result.patch:
        patch_preview = result.patch[:1000]
        if len(result.patch) > 1000:
            patch_preview += f"\n... ({len(result.patch) - 1000} more chars)"
        console.print(f"\n[bold]Patch ({len(result.patch)} chars):[/bold]")
        console.print(Panel(patch_preview, style="dim"))
    else:
        console.print("\n[yellow]No patch generated[/yellow]")


def _display_evaluation_result(console: Console, result: TestRunResult) -> None:
    """Display evaluation results."""
    if result.resolved is True:
        console.print("[bold green]✓ RESOLVED[/bold green] - Patch fixes the issue!")
    elif result.resolved is False:
        console.print("[bold red]✗ NOT RESOLVED[/bold red] - Patch does not fix the issue")
    else:
        console.print("[yellow]? UNKNOWN[/yellow] - Evaluation could not complete")

    if result.eval_error:
        console.print(f"  [dim]Error: {result.eval_error}[/dim]")

    console.print(f"  [dim]Evaluation time: {result.eval_duration_sec:.2f}s[/dim]")


def _display_summary(console: Console, result: TestRunResult) -> None:
    """Display final summary."""
    console.print()

    # Build summary table
    table = Table(title="Test Run Summary", show_header=False)
    table.add_column("", style="cyan", width=20)
    table.add_column("")

    table.add_row("Instance", result.instance_id)

    status = "[green]Success[/green]" if result.success else "[red]Failed[/red]"
    table.add_row("Agent Status", status)

    table.add_row("Duration", f"{result.duration_sec:.2f}s")
    table.add_row("Cost", f"${result.cost_usd:.4f}")
    table.add_row("Patch Generated", "Yes" if result.patch else "No")

    if result.resolved is not None:
        resolved = "[green]Yes[/green]" if result.resolved else "[red]No[/red]"
        table.add_row("Resolved", resolved)
    else:
        table.add_row("Resolved", "[dim]N/A[/dim]")

    console.print(table)


async def _clone_and_checkout(
    instance: SWEBenchInstance,
    work_dir: Path,
    console: Console,
) -> bool:
    """Clone repository and checkout base commit.

    Uses asyncio.to_thread() to avoid blocking the event loop during git operations.
    Returns True if successful, False otherwise.
    """
    repo_url = f"{instance.repo_url}.git"
    console.print(f"  Cloning [dim]{repo_url}[/dim]...")

    try:
        # Full clone (SWE-bench commits can be very old)
        # Run subprocess in thread to avoid blocking event loop
        result = await asyncio.to_thread(
            partial(
                subprocess.run,
                ["git", "clone", repo_url, str(work_dir)],
                capture_output=True,
                text=True,
                timeout=600,
            )
        )

        if result.returncode != 0:
            console.print(f"  [red]Clone failed:[/red] {result.stderr}")
            return False

        console.print("  [green]Clone successful[/green]")

        # Checkout base commit
        console.print(f"  Checking out [dim]{instance.base_commit[:12]}...[/dim]")

        result = await asyncio.to_thread(
            partial(
                subprocess.run,
                ["git", "checkout", instance.base_commit],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
        )

        if result.returncode != 0:
            console.print(f"  [red]Checkout failed:[/red] {result.stderr}")
            return False

        console.print("  [green]Checkout successful[/green]")
        return True

    except subprocess.TimeoutExpired:
        console.print("  [red]Git operation timed out[/red]")
        return False
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return False
