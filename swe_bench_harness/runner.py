"""Benchmark orchestration and execution.

This module coordinates the execution of benchmark runs across
plugin configurations and SWE-bench instances.
"""

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from swe_bench_harness.agent import ClaudeAgent
from swe_bench_harness.config import ExperimentConfig, PluginConfig
from swe_bench_harness.dataset import SWEBenchInstance
from swe_bench_harness.metrics import (
    BenchmarkResults,
    MetricsAggregator,
    RunRecord,
)


@dataclass
class ProgressEvent:
    """Event emitted during benchmark execution for progress tracking."""

    completed: int
    total: int
    current_instance: str
    current_config: str
    elapsed_sec: float


class BenchmarkRunner:
    """Orchestrate benchmark execution across configs and instances.

    Handles:
    - Iteration over plugin configs × instances × runs
    - Concurrent execution with semaphore
    - Progress event emission
    - Checkpoint save/restore for resume capability
    """

    def __init__(
        self,
        config: ExperimentConfig,
        instances: list[SWEBenchInstance],
        checkpoint_dir: Path | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            config: Experiment configuration
            instances: SWE-bench instances to benchmark
            checkpoint_dir: Directory for checkpoints (optional)
        """
        self.config = config
        self.instances = instances
        self.checkpoint_dir = checkpoint_dir
        self.records: list[RunRecord] = []
        self._start_time: float | None = None
        self._aggregator = MetricsAggregator(config.pricing)

    @property
    def total_runs(self) -> int:
        """Total number of runs to execute.

        Returns:
            configs × instances × runs_per_instance
        """
        return (
            len(self.config.configs)
            * len(self.instances)
            * self.config.execution.runs_per_instance
        )

    async def run(
        self,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> AsyncIterator[RunRecord]:
        """Execute benchmark with semaphore-based concurrency.

        Args:
            on_progress: Optional callback for progress updates

        Yields:
            RunRecord for each completed run
        """
        self._start_time = time.perf_counter()
        completed = len(self.records)  # Count already-completed runs from checkpoint

        semaphore = asyncio.Semaphore(self.config.execution.max_parallel_tasks)

        # Get already-completed runs for resume
        completed_runs = self._get_completed_run_keys()

        # Build tasks only for runs not yet completed
        tasks = []
        for plugin_config in self.config.configs:
            for instance in self.instances:
                for run_num in range(self.config.execution.runs_per_instance):
                    run_key = (instance.instance_id, plugin_config.id, run_num)
                    if run_key in completed_runs:
                        continue  # Skip already-completed runs

                    task = self._create_task(
                        semaphore=semaphore,
                        instance=instance,
                        plugin_config=plugin_config,
                        run_num=run_num,
                    )
                    tasks.append(task)

        # Execute tasks and yield results as they complete
        for coro in asyncio.as_completed(tasks):
            record = await coro
            self.records.append(record)
            completed += 1

            # Emit progress event
            if on_progress:
                elapsed = time.perf_counter() - self._start_time
                event = ProgressEvent(
                    completed=completed,
                    total=self.total_runs,
                    current_instance=record.instance_id,
                    current_config=record.config_id,
                    elapsed_sec=elapsed,
                )
                on_progress(event)

            # Save checkpoint periodically
            if self.checkpoint_dir and completed % 10 == 0:
                self.save_checkpoint(self.checkpoint_dir / "checkpoint.json")

            yield record

    async def _create_task(
        self,
        semaphore: asyncio.Semaphore,
        instance: SWEBenchInstance,
        plugin_config: PluginConfig,
        run_num: int,
    ) -> RunRecord:
        """Create and execute a single benchmark task.

        Args:
            semaphore: Concurrency limiter
            instance: SWE-bench instance
            plugin_config: Plugin configuration
            run_num: Run number (for repeated runs)

        Returns:
            RunRecord with execution results
        """
        async with semaphore:
            return await self._run_single(
                instance=instance,
                plugin_config=plugin_config,
                run_num=run_num,
            )

    async def _run_single(
        self,
        instance: SWEBenchInstance,
        plugin_config: PluginConfig,
        run_num: int,
    ) -> RunRecord:
        """Execute a single benchmark run.

        Args:
            instance: SWE-bench instance
            plugin_config: Plugin configuration
            run_num: Run number

        Returns:
            RunRecord with execution results
        """
        run_id = f"{instance.instance_id}_{plugin_config.id}_run{run_num}_{uuid.uuid4().hex[:8]}"

        # Create agent
        agent = ClaudeAgent(
            model_config=self.config.model,
            plugin_config=plugin_config,
        )

        # Execute
        result = await agent.execute(
            instance=instance,
            timeout_sec=self.config.execution.timeout_per_run_sec,
        )

        # Calculate cost
        cost = self._aggregator.calculate_cost(
            tokens_input=result.tokens_input,
            tokens_output=result.tokens_output,
            tokens_cache_read=result.tokens_cache_read,
        )

        # Create record
        return RunRecord(
            run_id=run_id,
            instance_id=instance.instance_id,
            config_id=plugin_config.id,
            timestamp=datetime.now(),
            success=result.success,
            failure_type=result.failure_type,
            duration_sec=result.duration_sec,
            tokens_input=result.tokens_input,
            tokens_output=result.tokens_output,
            tokens_cache_read=result.tokens_cache_read,
            tool_calls_total=result.tool_calls_total,
            tool_calls_by_name=result.tool_calls_by_name,
            error_reason=result.error_reason,
            cost_usd=cost,
            patch_generated=result.patch_generated,
        )

    def save_checkpoint(self, path: Path) -> None:
        """Save current records to JSON for resume capability.

        Args:
            path: Path to checkpoint file
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "experiment_name": self.config.name,
            "timestamp": datetime.now().isoformat(),
            "completed_runs": len(self.records),
            "total_runs": self.total_runs,
            "records": [r.to_dict() for r in self.records],
        }

        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, path: Path) -> int:
        """Load records from checkpoint for resume.

        Args:
            path: Path to checkpoint file

        Returns:
            Number of records loaded
        """
        if not path.exists():
            return 0

        with open(path) as f:
            checkpoint = json.load(f)

        self.records = [
            RunRecord.from_dict(r) for r in checkpoint.get("records", [])
        ]

        return len(self.records)

    def get_results(self) -> BenchmarkResults:
        """Aggregate records into final benchmark results.

        Returns:
            BenchmarkResults with all records and summaries
        """
        total_duration = (
            time.perf_counter() - self._start_time if self._start_time else 0.0
        )

        summaries = self._aggregator.aggregate(
            records=self.records,
            configs=self.config.configs,
        )

        return BenchmarkResults(
            experiment_name=self.config.name,
            timestamp=datetime.now(),
            total_duration_sec=total_duration,
            records=self.records,
            summaries=summaries,
        )

    def _get_completed_run_keys(self) -> set[tuple[str, str, int]]:
        """Get keys of completed runs for resume filtering.

        Returns:
            Set of (instance_id, config_id, run_num) tuples
        """
        completed = set()
        for record in self.records:
            # Extract run_num from run_id - format: {instance}_{config}_run{N}_{uuid}
            # Use the stored instance_id and config_id directly
            run_num = self._extract_run_num(record.run_id)
            if run_num is not None:
                completed.add((record.instance_id, record.config_id, run_num))
        return completed

    @staticmethod
    def _extract_run_num(run_id: str) -> int | None:
        """Extract run number from run ID.

        Args:
            run_id: Run ID in format {instance}_{config}_run{N}_{uuid}

        Returns:
            Run number or None if parsing fails
        """
        match = re.search(r"_run(\d+)_", run_id)
        if match:
            return int(match.group(1))
        return None

    def get_completed_run_ids(self) -> set[tuple[str, str, int]]:
        """Get IDs of completed runs for resume filtering.

        Returns:
            Set of (instance_id, config_id, run_num) tuples
        """
        return self._get_completed_run_keys()


async def run_benchmark(
    config: ExperimentConfig,
    instances: list[SWEBenchInstance],
    on_progress: Callable[[ProgressEvent], None] | None = None,
    checkpoint_dir: Path | None = None,
) -> BenchmarkResults:
    """Convenience function to run a complete benchmark.

    Args:
        config: Experiment configuration
        instances: SWE-bench instances
        on_progress: Optional progress callback
        checkpoint_dir: Optional checkpoint directory

    Returns:
        Complete BenchmarkResults
    """
    runner = BenchmarkRunner(
        config=config,
        instances=instances,
        checkpoint_dir=checkpoint_dir,
    )

    async for _ in runner.run(on_progress=on_progress):
        pass  # Process all runs

    return runner.get_results()
