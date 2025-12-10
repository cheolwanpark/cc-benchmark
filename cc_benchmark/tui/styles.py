"""Rich styles and themes for cc-benchmark TUI.

This module centralizes all styling constants for consistent theming
across the TUI application.
"""

from rich.theme import Theme

# Named style constants for easy reference
STYLES = {
    # Panels
    "welcome": "bold blue",
    "wizard_header": "bold cyan",
    "progress_panel": "blue",
    "error_panel": "bold red",
    "success_panel": "bold green",
    "summary_panel": "green",
    # Text
    "heading": "bold white",
    "subheading": "bold cyan",
    "label": "cyan",
    "value": "green",
    "value_dim": "dim green",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    # Progress
    "progress_complete": "green",
    "progress_pending": "blue",
    "progress_failed": "red",
    # Stats
    "stat_label": "cyan",
    "stat_value": "white",
    "stat_good": "green",
    "stat_bad": "red",
    "stat_cost": "yellow",
    # Wizard steps
    "step_number": "bold cyan",
    "step_description": "dim",
}

# Custom theme for Rich console
CC_BENCHMARK_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "progress.percentage": "green",
    "progress.elapsed": "cyan",
    "progress.remaining": "cyan dim",
    "bar.complete": "green",
    "bar.finished": "bright_green",
    "bar.pulse": "blue",
})


def get_status_style(success: bool, resolved: bool = False) -> str:
    """Get style for a record status.

    Args:
        success: Whether the run was successful
        resolved: Whether the issue was resolved

    Returns:
        Style string for the status
    """
    if resolved:
        return STYLES["success"]
    elif success:
        return STYLES["value"]
    else:
        return STYLES["error"]


def format_cost(cost_usd: float) -> str:
    """Format cost with appropriate styling.

    Args:
        cost_usd: Cost in USD

    Returns:
        Formatted cost string with Rich markup
    """
    if cost_usd < 0.10:
        style = "green"
    elif cost_usd < 1.00:
        style = "yellow"
    else:
        style = "red"
    return f"[{style}]${cost_usd:.2f}[/{style}]"


def format_rate(rate: float) -> str:
    """Format a rate (0.0-1.0) as percentage with color.

    Args:
        rate: Rate value between 0.0 and 1.0

    Returns:
        Formatted percentage string with Rich markup
    """
    percent = rate * 100
    if percent >= 50:
        style = "green"
    elif percent >= 25:
        style = "yellow"
    else:
        style = "red"
    return f"[{style}]{percent:.1f}%[/{style}]"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
