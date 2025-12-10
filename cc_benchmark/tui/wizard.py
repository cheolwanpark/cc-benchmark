"""Interactive configuration wizard for cc-benchmark.

This module provides a step-by-step wizard for creating benchmark
configurations interactively using Rich prompts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from cc_benchmark.config import (
    DATASET_SOURCES,
    Config,
    DatasetConfig,
    ExecutionConfig,
)
from cc_benchmark.tui.styles import STYLES

if TYPE_CHECKING:
    pass


class ConfigWizard:
    """Interactive wizard for creating benchmark configuration.

    Guides the user through a series of prompts to build a complete
    benchmark configuration, with validation at each step.
    """

    # Available models
    MODELS = ["claude-sonnet-4-5", "claude-opus-4-5"]

    # Parallelism bounds
    MIN_PARALLELISM = 1
    MAX_PARALLELISM = 16
    DEFAULT_PARALLELISM = 4

    # Other defaults
    DEFAULT_TIMEOUT = 900
    DEFAULT_OUTPUT_DIR = "./results"
    DEFAULT_DOCKER_MEMORY = "8g"
    DEFAULT_DOCKER_CPUS = 4

    def __init__(self, console: Console):
        """Initialize the wizard.

        Args:
            console: Rich Console for output
        """
        self.console = console
        self._config: Config | None = None

    def _print_step(self, step: str, title: str, description: str | None = None) -> None:
        """Print step header with optional description."""
        sn = STYLES["step_number"]
        self.console.print(f"\n[{sn}]{step}:[/{sn}] {title}")
        if description:
            sd = STYLES["step_description"]
            self.console.print(f"[{sd}]{description}[/{sd}]")

    def run(self) -> Config | None:
        """Run the interactive wizard flow.

        Note: This method is synchronous as all Rich prompts are blocking.

        Returns:
            Config object if completed successfully, None if cancelled.
        """
        self.console.print()
        self.console.print(Panel(
            "[bold]Interactive Configuration Wizard[/bold]\n"
            "[dim]Press Ctrl+C at any time to cancel[/dim]",
            style=STYLES["wizard_header"],
        ))

        try:
            # Core options (always shown)
            name = self._prompt_experiment_name()
            dataset_name = self._prompt_dataset()
            split = self._prompt_split()
            model = self._prompt_model()
            max_parallel = self._prompt_parallelism()
            timeout_sec = self._prompt_timeout()
            output_dir = self._prompt_output_dir()

            # Start building execution config with defaults
            docker_memory = self.DEFAULT_DOCKER_MEMORY
            docker_cpus = self.DEFAULT_DOCKER_CPUS
            plugins: list[str] = []

            # Advanced options (optional)
            if self._prompt_advanced_options():
                docker_memory = self._prompt_docker_memory()
                docker_cpus = self._prompt_docker_cpus()
                plugins = self._prompt_plugins()

            # Build config
            self._config = Config(
                name=name,
                dataset=DatasetConfig(name=dataset_name, split=split),
                execution=ExecutionConfig(
                    max_parallel=max_parallel,
                    timeout_sec=timeout_sec,
                    docker_memory=docker_memory,
                    docker_cpus=docker_cpus,
                ),
                model=model,
                plugins=plugins,
                output_dir=output_dir,
            )

            # Display summary
            self._display_summary()

            # Confirm
            if not Confirm.ask(
                "\n[bold]Proceed with this configuration?[/bold]",
                default=True
            ):
                self.console.print("[yellow]Configuration cancelled.[/yellow]")
                return None

            return self._config

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Wizard cancelled.[/yellow]")
            return None

    def _prompt_experiment_name(self) -> str:
        """Step 1: Prompt for experiment name."""
        self._print_step("Step 1/7", "Experiment Name", "A descriptive name for this run")

        while True:
            name = Prompt.ask("  Name", default="my-benchmark")
            if name.strip():
                return name.strip()
            self.console.print("[red]Name cannot be empty. Try again.[/red]")

    def _prompt_dataset(self) -> str:
        """Step 2: Prompt for dataset selection."""
        self._print_step("Step 2/7", "Dataset Selection")

        # Show options table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Option", width=8)
        table.add_column("Name", width=10)
        table.add_column("HuggingFace Path", width=35)
        table.add_column("Description", width=25)

        table.add_row("1", "lite", DATASET_SOURCES["lite"], "SWE-bench Lite (300)")
        table.add_row("2", "full", DATASET_SOURCES["full"], "Full SWE-bench (2,294)")
        table.add_row("3", "verified", DATASET_SOURCES["verified"], "SWE-bench Verified (500)")

        self.console.print(table)

        choice = Prompt.ask(
            "  Select dataset",
            choices=["1", "2", "3", "lite", "full", "verified"],
            default="1",
        )

        mapping = {"1": "lite", "2": "full", "3": "verified"}
        return mapping.get(choice, choice)

    def _prompt_split(self) -> str:
        """Step 3: Prompt for dataset split."""
        self._print_step("Step 3/7", "Dataset Split", "Python slice notation (e.g., :10)")
        self.console.print("  Examples:")
        self.console.print("    [dim]:10[/dim]      First 10 instances")
        self.console.print("    [dim]10:20[/dim]    Instances 10-19")
        self.console.print("    [dim]20:[/dim]      From index 20 to end")
        self.console.print("    [dim]:[/dim]        All instances")

        # Regex for valid split patterns
        split_pattern = re.compile(r"^:?\d*:?\d*$")

        while True:
            split = Prompt.ask("  Split", default=":10")
            if split_pattern.match(split):
                return split
            self.console.print("[red]Invalid split format. Use :10, 10:20, etc.[/red]")

    def _prompt_model(self) -> str:
        """Step 4: Prompt for model selection."""
        self._print_step("Step 4/7", "Claude Model")

        for i, model in enumerate(self.MODELS, 1):
            self.console.print(f"  {i}. {model}")

        choice = Prompt.ask(
            "  Select model",
            choices=["1", "2"] + self.MODELS,
            default="1",
        )

        if choice in ["1", "2"]:
            return self.MODELS[int(choice) - 1]
        return choice

    def _prompt_parallelism(self) -> int:
        """Step 5: Prompt for parallelism."""
        desc = f"Number of concurrent runs ({self.MIN_PARALLELISM}-{self.MAX_PARALLELISM})"
        self._print_step("Step 5/7", "Parallelism", desc)

        while True:
            value = IntPrompt.ask("  Max parallel", default=self.DEFAULT_PARALLELISM)
            if self.MIN_PARALLELISM <= value <= self.MAX_PARALLELISM:
                return value
            min_p, max_p = self.MIN_PARALLELISM, self.MAX_PARALLELISM
            self.console.print(f"[red]Must be {min_p}-{max_p}. Try again.[/red]")

    def _prompt_timeout(self) -> int:
        """Step 6: Prompt for timeout."""
        self._print_step("Step 6/7", "Timeout", "Maximum seconds per instance")

        while True:
            value = IntPrompt.ask("  Timeout (seconds)", default=self.DEFAULT_TIMEOUT)
            if value >= 60:
                return value
            self.console.print("[red]Timeout must be at least 60 seconds. Try again.[/red]")

    def _prompt_output_dir(self) -> str:
        """Step 7: Prompt for output directory."""
        self._print_step("Step 7/7", "Output Directory", "Directory for results")

        while True:
            value = Prompt.ask("  Output dir", default=self.DEFAULT_OUTPUT_DIR)
            path = Path(value).expanduser()

            # Try to create or verify directory
            try:
                path.mkdir(parents=True, exist_ok=True)
                # Check if writeable by trying to create a temp file
                test_file = path / ".test_write"
                test_file.touch()
                test_file.unlink()
                return value
            except PermissionError:
                self.console.print(f"[red]Cannot write to '{value}'. Check permissions.[/red]")
            except Exception as e:
                self.console.print(f"[red]Invalid path '{value}': {e}[/red]")

    def _prompt_advanced_options(self) -> bool:
        """Ask if user wants to configure advanced options."""
        self.console.print()
        return Confirm.ask(
            "[bold]Configure advanced options?[/bold] (Docker settings, plugins)",
            default=False,
        )

    def _prompt_docker_memory(self) -> str:
        """Prompt for Docker memory limit."""
        self._print_step("Advanced", "Docker Memory", "e.g., '4g', '8g', '16g'")

        # Regex for valid memory format
        memory_pattern = re.compile(r"^\d+[gmGM]$")

        while True:
            value = Prompt.ask("  Memory limit", default=self.DEFAULT_DOCKER_MEMORY)
            if memory_pattern.match(value):
                return value.lower()
            self.console.print("[red]Invalid format. Use format like '4g', '8g', '16g'.[/red]")

    def _prompt_docker_cpus(self) -> int:
        """Prompt for Docker CPU limit."""
        self._print_step("Advanced", "Docker CPUs", "Container CPU limit")

        while True:
            value = IntPrompt.ask("  CPU limit", default=self.DEFAULT_DOCKER_CPUS)
            if 1 <= value <= 32:
                return value
            self.console.print("[red]Must be between 1 and 32. Try again.[/red]")

    def _prompt_plugins(self) -> list[str]:
        """Prompt for plugin paths."""
        self._print_step("Advanced", "Plugin Paths", "Enter paths, empty to finish")

        plugins = []
        while True:
            value = Prompt.ask("  Plugin path (empty to finish)", default="")
            if not value:
                break

            # Validate path
            path = Path(value).expanduser()
            if not path.exists():
                self.console.print(f"[yellow]Warning: Path '{value}' does not exist[/yellow]")
                if not Confirm.ask("  Add anyway?", default=False):
                    continue

            # Ensure path format is valid for config
            if not value.startswith(("./", "/", "~")):
                value = f"./{value}"

            plugins.append(value)
            self.console.print(f"  [green]Added:[/green] {value}")

        return plugins

    def _display_summary(self) -> None:
        """Display configuration summary table."""
        if self._config is None:
            return

        table = Table(title="Configuration Summary", show_header=False)
        table.add_column("Setting", style="cyan", width=20)
        table.add_column("Value", style="green")

        table.add_row("Experiment Name", self._config.name)
        table.add_row("Dataset", self._config.dataset.name)
        table.add_row("Split", self._config.dataset.split)
        table.add_row("Model", self._config.model)
        table.add_row("Max Parallel", str(self._config.execution.max_parallel))
        table.add_row("Timeout", f"{self._config.execution.timeout_sec}s")
        table.add_row("Output Directory", self._config.output_dir)

        # Show advanced options if non-default
        if self._config.execution.docker_memory != self.DEFAULT_DOCKER_MEMORY:
            table.add_row("Docker Memory", self._config.execution.docker_memory)
        if self._config.execution.docker_cpus != self.DEFAULT_DOCKER_CPUS:
            table.add_row("Docker CPUs", str(self._config.execution.docker_cpus))
        if self._config.plugins:
            table.add_row("Plugins", ", ".join(self._config.plugins))

        self.console.print("\n")
        self.console.print(table)

    def prompt_save_config(self) -> bool:
        """Ask if user wants to save config to YAML file."""
        return Confirm.ask(
            "\n[bold]Save configuration to YAML file?[/bold]",
            default=False,
        )

    def prompt_filename(self) -> str:
        """Prompt for config filename."""
        while True:
            filename = Prompt.ask("  Filename", default="config.yaml")
            if filename.endswith((".yaml", ".yml")):
                return filename
            self.console.print("[yellow]Adding .yaml extension[/yellow]")
            return f"{filename}.yaml"

    def prompt_test_run(self) -> bool:
        """Ask if user wants to run single-instance test first."""
        return Confirm.ask(
            "[bold]Run test on single instance first?[/bold]",
            default=False,
        )

    def prompt_continue_after_test_failure(self) -> bool:
        """Ask if user wants to continue after test run failure."""
        return Confirm.ask(
            "[yellow]Test run had issues. Continue with full benchmark?[/yellow]",
            default=False,
        )
