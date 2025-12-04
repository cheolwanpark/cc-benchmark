"""Benchmark orchestration and execution.

This module coordinates the execution of benchmark runs across
plugin configurations and SWE-bench instances.
"""

import asyncio
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from swe_bench_harness.agent import ClaudeAgent
from swe_bench_harness.config import BenchmarkConfig, ExperimentConfig
from swe_bench_harness.dataset import SWEBenchInstance
from swe_bench_harness.metrics import (
    BenchmarkResults,
    MetricsAggregator,
    RunRecord,
)
from swe_bench_harness.plugins import plugin_context


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
    - Iteration over benchmark configs × instances × runs
    - Concurrent execution with semaphore
    - Progress event emission
    - Checkpoint save/restore for resume capability
    - Plugin loading and cleanup
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
        self._aggregator = MetricsAggregator()
        self._resolved_plugins: dict[str, list[dict[str, Any]]] = {}

    @property
    def total_runs(self) -> int:
        """Total number of runs to execute.

        Returns:
            configs × instances × runs
        """
        return (
            len(self.config.configs)
            * len(self.instances)
            * self.config.execution.runs
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

        # Use plugin context to manage plugin loading and cleanup
        with plugin_context() as loader:
            # Resolve all plugin URIs to local paths
            for benchmark_config in self.config.configs:
                self._resolved_plugins[benchmark_config.name] = [
                    {"type": "local", "path": loader.load(p.uri)}
                    for p in benchmark_config.plugins
                ]

            semaphore = asyncio.Semaphore(self.config.execution.max_parallel)

            # Get already-completed runs for resume
            completed_runs = self._get_completed_run_keys()

            # Build tasks only for runs not yet completed
            tasks = []
            for benchmark_config in self.config.configs:
                for instance in self.instances:
                    for run_num in range(self.config.execution.runs):
                        run_key = (instance.instance_id, benchmark_config.name, run_num)
                        if run_key in completed_runs:
                            continue  # Skip already-completed runs

                        task = self._create_task(
                            semaphore=semaphore,
                            instance=instance,
                            benchmark_config=benchmark_config,
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
        benchmark_config: BenchmarkConfig,
        run_num: int,
    ) -> RunRecord:
        """Create and execute a single benchmark task.

        Args:
            semaphore: Concurrency limiter
            instance: SWE-bench instance
            benchmark_config: Benchmark configuration
            run_num: Run number (for repeated runs)

        Returns:
            RunRecord with execution results
        """
        async with semaphore:
            return await self._run_single(
                instance=instance,
                benchmark_config=benchmark_config,
                run_num=run_num,
            )

    async def _run_single(
        self,
        instance: SWEBenchInstance,
        benchmark_config: BenchmarkConfig,
        run_num: int,
    ) -> RunRecord:
        """Execute a single benchmark run.

        Args:
            instance: SWE-bench instance
            benchmark_config: Benchmark configuration
            run_num: Run number

        Returns:
            RunRecord with execution results
        """
        from swe_bench_harness.agent import ExecutionResult
        from swe_bench_harness.metrics import FailureType

        run_id = f"{instance.instance_id}_{benchmark_config.name}_run{run_num}_{uuid.uuid4().hex[:8]}"
        work_dir: Path | None = None
        start_time = time.perf_counter()
        resolved = False  # Will be set to True only if patch passes SWE-bench evaluation

        try:
            # Prepare working directory with cloned repo
            work_dir = await self._prepare_work_dir(instance)

            # Get resolved plugins for this config
            resolved_plugins = self._resolved_plugins.get(benchmark_config.name, [])

            # Create agent
            agent = ClaudeAgent(
                model_config=self.config.model,
                benchmark_config=benchmark_config,
                resolved_plugins=resolved_plugins,
            )

            # Execute with working directory
            result = await agent.execute(
                instance=instance,
                timeout_sec=self.config.execution.timeout_sec,
                cwd=work_dir,
            )

            # Run SWE-bench evaluation if patch was generated
            if result.patch_generated:
                resolved = await self._evaluate_patch(instance, result.patch_generated)

        except subprocess.CalledProcessError as e:
            # Git clone or checkout failed
            error_msg = e.stderr.decode() if e.stderr else str(e)
            result = ExecutionResult(
                success=False,
                failure_type=FailureType.UNKNOWN,
                duration_sec=time.perf_counter() - start_time,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason=f"Git setup failed: {error_msg}",
            )

        except subprocess.TimeoutExpired as e:
            # Git operation timed out
            result = ExecutionResult(
                success=False,
                failure_type=FailureType.TIMEOUT,
                duration_sec=time.perf_counter() - start_time,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason=f"Git setup timed out after {e.timeout}s",
            )

        except Exception as e:
            # Unexpected error during setup
            result = ExecutionResult(
                success=False,
                failure_type=FailureType.UNKNOWN,
                duration_sec=time.perf_counter() - start_time,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason=f"Setup error: {e}",
            )

        finally:
            # Clean up temp directory (remove parent dir which contains repo/)
            if work_dir and work_dir.parent.exists():
                shutil.rmtree(work_dir.parent, ignore_errors=True)

        # Create record (cost comes from agent result)
        return RunRecord(
            run_id=run_id,
            instance_id=instance.instance_id,
            config_id=benchmark_config.name,
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
            cost_usd=result.cost_usd,
            patch_generated=result.patch_generated,
            resolved=resolved,
        )

    async def _prepare_work_dir(self, instance: SWEBenchInstance) -> Path:
        """Prepare working directory with cloned repo at base commit.

        Args:
            instance: SWE-bench instance with repo info

        Returns:
            Path to temporary directory with repo clone

        Raises:
            subprocess.CalledProcessError: If git operations fail
            subprocess.TimeoutExpired: If git operations time out
        """
        # Create base temp directory
        base_dir = Path(tempfile.mkdtemp(prefix=f"swe_bench_{instance.instance_id}_"))
        work_dir = base_dir / "repo"

        # Full clone (shallow clone often misses old base commits)
        # Use 10 minute timeout for large repos
        repo_url = instance.repo_url
        await asyncio.to_thread(
            subprocess.run,
            ["git", "clone", repo_url, str(work_dir)],
            capture_output=True,
            check=True,
            timeout=600,  # 10 minute timeout for clone
        )

        # Checkout base commit (1 minute timeout)
        await asyncio.to_thread(
            subprocess.run,
            ["git", "checkout", instance.base_commit],
            cwd=work_dir,
            capture_output=True,
            check=True,
            timeout=60,  # 1 minute timeout for checkout
        )

        return work_dir

    async def _evaluate_patch(
        self,
        instance: SWEBenchInstance,
        patch: str,
    ) -> bool:
        """Run official SWE-bench evaluation on patch.

        Uses Docker to run the test suite and verify the patch
        fixes the issue (Fail-to-Pass tests pass, Pass-to-Pass tests still pass).

        Args:
            instance: SWE-bench instance
            patch: Generated patch (unified diff format)

        Returns:
            True if patch resolves the issue, False otherwise
        """
        try:
            import docker
            from swebench.harness.run_evaluation import run_instance
            from swebench.harness.test_spec.test_spec import make_test_spec
        except ImportError as e:
            logger.warning(f"SWE-bench evaluation unavailable: {e}")
            return False

        try:
            # Get Docker client
            client = docker.from_env()

            # Create test spec from instance dict
            test_spec = make_test_spec(instance.to_dict())

            # Create prediction dict
            prediction = {
                "instance_id": instance.instance_id,
                "model_name_or_path": self.config.model.name,
                "model_patch": patch,
            }

            # Run evaluation in thread pool with timeout enforcement
            # run_instance signature: (test_spec, pred, rm_image, force_rebuild, client, run_id, timeout, rewrite_reports)
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    run_instance,
                    test_spec,
                    prediction,
                    False,  # rm_image - keep images for reuse
                    False,  # force_rebuild
                    client,
                    "benchmark",  # run_id
                    self.config.execution.eval_timeout,
                ),
                timeout=self.config.execution.eval_timeout + 60,  # Extra buffer for cleanup
            )

            # Check if resolved
            if result is None:
                logger.warning(f"run_instance returned None for {instance.instance_id}")
                return False
            resolved = result.get("resolved", False)
            if "resolved" not in result:
                logger.warning(f"Missing 'resolved' key in result for {instance.instance_id}: {result.keys()}")
            logger.debug(
                f"Evaluation result for {instance.instance_id}: resolved={resolved}"
            )
            return resolved

        except asyncio.TimeoutError:
            logger.warning(f"Evaluation timed out for {instance.instance_id}")
            return False
        except docker.errors.DockerException as e:
            logger.warning(f"Docker error during evaluation: {e}")
            return False
        except Exception as e:
            logger.warning(f"Evaluation failed for {instance.instance_id}: {e}")
            return False

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
