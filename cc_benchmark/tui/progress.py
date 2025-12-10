"""Progress display components for benchmark execution.

This module provides Rich-based progress bars and live dashboard
components for tracking benchmark execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from cc_benchmark.metrics import RunRecord
from cc_benchmark.tui.styles import STYLES, format_cost, format_rate


@dataclass
class BenchmarkStats:
    """Live statistics for benchmark execution.

    Tracks progress metrics during benchmark runs and provides
    computed properties for rates and timing.
    """

    total: int = 0
    completed: int = 0
    patches_generated: int = 0
    resolved: int = 0
    failed: int = 0
    total_cost_usd: float = 0.0
    current_instance: str = ""
    start_time: datetime = field(default_factory=datetime.now)

    def update(self, record: RunRecord) -> None:
        """Update stats from a completed record.

        Args:
            record: The completed RunRecord to update stats from
        """
        self.completed += 1
        if record.patch:
            self.patches_generated += 1
        if record.resolved:
            self.resolved += 1
        if not record.success:
            self.failed += 1
        self.total_cost_usd += record.cost_usd

    @property
    def patch_rate(self) -> float:
        """Calculate patch generation rate."""
        if self.completed == 0:
            return 0.0
        return self.patches_generated / self.completed

    @property
    def resolve_rate(self) -> float:
        """Calculate resolution rate."""
        if self.completed == 0:
            return 0.0
        return self.resolved / self.completed

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

    def reset(self, total: int) -> None:
        """Reset stats for a new benchmark run.

        Args:
            total: Total number of instances to process
        """
        self.total = total
        self.completed = 0
        self.patches_generated = 0
        self.resolved = 0
        self.failed = 0
        self.total_cost_usd = 0.0
        self.current_instance = ""
        self.start_time = datetime.now()


class BenchmarkProgress:
    """Rich-based progress display with live stats dashboard.

    This class manages a Rich Live display that shows:
    - Progress bar with completion count and ETA
    - Live statistics (patches, resolved, failed, cost)
    - Current instance being processed

    Usage:
        progress = BenchmarkProgress(console, total_instances)
        with progress:
            async for record in run_benchmark(..., on_progress=progress.on_progress):
                progress.update(record)
    """

    def __init__(self, console: Console, total: int):
        """Initialize progress display.

        Args:
            console: Rich Console for output
            total: Total number of instances to process
        """
        self.console = console
        self.stats = BenchmarkStats(total=total, start_time=datetime.now())

        # Progress bar configuration
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("["),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
            TextColumn("]"),
            console=console,
            expand=True,
        )

        self._main_task = None
        self._live: Live | None = None

    def __enter__(self):
        """Start the live display."""
        self._main_task = self.progress.add_task(
            "Processing instances",
            total=self.stats.total,
        )

        self._live = Live(
            self._create_layout(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the live display."""
        if self._live:
            self._live.stop()
        return False

    def _create_layout(self) -> Panel:
        """Create the combined progress + stats layout."""
        return Panel(
            Group(
                self.progress,
                "",  # Spacer
                self._create_stats_table(),
            ),
            title="[bold]Benchmark Progress[/bold]",
            border_style=STYLES["progress_panel"],
            padding=(1, 2),
        )

    def _create_stats_table(self) -> Table:
        """Create the live stats table."""
        stats = self.stats

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="cyan", width=20)
        table.add_column(justify="left", width=15)
        table.add_column(justify="right", style="cyan", width=20)
        table.add_column(justify="left", width=15)

        # Row 1: Patches and Resolved
        table.add_row(
            "Patches Generated:",
            f"[green]{stats.patches_generated}[/green] ({format_rate(stats.patch_rate)})",
            "Resolved:",
            f"[green]{stats.resolved}[/green] ({format_rate(stats.resolve_rate)})",
        )

        # Row 2: Failed and Cost
        failed_style = "red" if stats.failed > 0 else "dim"
        table.add_row(
            "Failed:",
            f"[{failed_style}]{stats.failed}[/{failed_style}]",
            "Total Cost:",
            format_cost(stats.total_cost_usd),
        )

        # Row 3: Current instance
        if stats.current_instance:
            # Truncate long instance IDs
            instance_display = stats.current_instance
            if len(instance_display) > 40:
                instance_display = instance_display[:37] + "..."
            table.add_row(
                "Current:",
                f"[dim]{instance_display}[/dim]",
                "",
                "",
            )

        return table

    def on_progress(self, completed: int, total: int, instance_id: str) -> None:
        """Callback for runner progress updates.

        This method matches the signature expected by run_benchmark's
        on_progress parameter.

        Args:
            completed: Number of completed instances
            total: Total number of instances
            instance_id: ID of the just-completed or current instance
        """
        self.stats.current_instance = instance_id
        if self._main_task is not None:
            self.progress.update(self._main_task, completed=completed)
        self._refresh()

    def update(self, record: RunRecord) -> None:
        """Update display with completed record.

        Args:
            record: Completed RunRecord with results
        """
        self.stats.update(record)
        self._refresh()

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._create_layout())

    def stop(self) -> None:
        """Stop the live display (for error handling)."""
        if self._live:
            self._live.stop()
            self._live = None


class SimpleProgress:
    """Simpler progress display for test runs (single instance).

    Shows spinner + status text without full dashboard.
    Useful for single instance test runs where the full
    dashboard would be overkill.
    """

    def __init__(self, console: Console, description: str = "Running test..."):
        """Initialize simple progress.

        Args:
            console: Rich Console for output
            description: Initial status description
        """
        self.console = console
        self._description = description
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        )
        self._task = None
        self._live: Live | None = None

    def __enter__(self):
        """Start the live display."""
        self._task = self.progress.add_task(self._description, total=None)
        self._live = Live(self.progress, console=self.console)
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the live display."""
        if self._live:
            self._live.stop()
        return False

    def update_status(self, status: str) -> None:
        """Update the status text.

        Args:
            status: New status description
        """
        if self._task is not None:
            self.progress.update(self._task, description=status)

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None
