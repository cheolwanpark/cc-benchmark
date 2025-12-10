"""Benchmark orchestration and execution.

This module coordinates the execution of benchmark runs across
SWE-bench instances with parallel processing.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path

from cc_benchmark.agent import run_agent
from cc_benchmark.config import Config
from cc_benchmark.dataset import SWEBenchInstance
from cc_benchmark.evaluation import evaluate
from cc_benchmark.metrics import FailureType, RunRecord


async def run_benchmark(
    config: Config,
    instances: list[SWEBenchInstance],
    on_progress: Callable[[int, int, str], None] | None = None,
    verbose: bool = False,
) -> AsyncIterator[RunRecord]:
    """Execute benchmark on all instances.

    Args:
        config: Benchmark configuration
        instances: SWE-bench instances to process
        on_progress: Optional callback(completed, total, instance_id)
        verbose: Whether to print agent responses to stderr (default: False)

    Yields:
        RunRecord for each completed instance
    """
    # Pull all unique evaluation images upfront
    await _pull_eval_images(instances, config)

    semaphore = asyncio.Semaphore(config.execution.max_parallel)
    completed = 0
    total = len(instances)

    async def process_with_semaphore(instance: SWEBenchInstance) -> RunRecord:
        async with semaphore:
            return await _process_instance(instance, config, verbose)

    # Create all tasks
    tasks = [
        asyncio.create_task(process_with_semaphore(inst))
        for inst in instances
    ]

    # Yield results as they complete
    for coro in asyncio.as_completed(tasks):
        record = await coro
        completed += 1

        if on_progress:
            on_progress(completed, total, record.instance_id)

        yield record


async def _pull_eval_images(
    instances: list[SWEBenchInstance],
    config: Config,
) -> None:
    """Pull Epoch AI evaluation images and tag them for SWE-bench compatibility.

    Epoch AI provides complete eval images that contain everything needed.
    However, SWE-bench expects a 3-layer structure: base → env → eval.

    We pull Epoch AI images and tag them as BOTH the env and eval images
    so SWE-bench can use them directly without trying to build.

    Epoch AI format: ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}
    SWE-bench expects:
      - Eval: sweb.eval.{arch}.{instance_id}:latest
      - Env: sweb.env.{lang}.{arch}.{hash}:latest (computed from test spec)
    """
    import platform

    from swebench.harness.test_spec.test_spec import make_test_spec

    # Get unique instances
    unique_instances = {inst.instance_id: inst for inst in instances}.values()

    # Determine architecture
    arch = "x86_64" if platform.machine() in ("x86_64", "AMD64") else "arm64"

    # Pull images in parallel with limited concurrency
    semaphore = asyncio.Semaphore(4)

    async def pull_and_tag(instance) -> None:
        async with semaphore:
            instance_id = instance.instance_id

            # Epoch AI image format
            remote_image = f"ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}"

            # SWE-bench eval image format
            eval_tag = f"sweb.eval.{arch}.{instance_id.lower()}:latest"

            # Compute the env image tag that SWE-bench will look for
            # We need to create a test spec to get the exact env_image_key
            test_spec = make_test_spec(
                instance.to_dict(),
                arch=arch,
                namespace=None,
            )
            env_tag = test_spec.env_image_key

            try:
                # Pull from Epoch AI registry
                await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "pull", remote_image],
                    capture_output=True,
                    timeout=600,  # 10 minute timeout
                )

                # Tag as eval image
                await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "tag", remote_image, eval_tag],
                    capture_output=True,
                    timeout=30,
                )

                # Tag as env image (so SWE-bench doesn't try to build it)
                await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "tag", remote_image, env_tag],
                    capture_output=True,
                    timeout=30,
                )
            except Exception:
                # Ignore failures - evaluation will try to build locally if needed
                pass

    await asyncio.gather(*[pull_and_tag(inst) for inst in unique_instances])


async def _process_instance(
    instance: SWEBenchInstance,
    config: Config,
    verbose: bool = False,
) -> RunRecord:
    """Process a single instance: clone repo, run agent, evaluate.

    Args:
        instance: SWE-bench instance to process
        config: Benchmark configuration
        verbose: Whether to print agent responses to stderr (default: False)

    Returns:
        RunRecord with all execution metrics
    """
    start_time = time.perf_counter()
    work_dir: Path | None = None

    try:
        # Clone repo and checkout base commit
        work_dir = await _prepare_workspace(instance)

        # Create output directory for agent
        output_dir = work_dir.parent / "output"
        output_dir.mkdir(exist_ok=True)

        # Run the agent
        agent_result = await run_agent(
            instance=instance,
            config=config,
            work_dir=work_dir,
            output_dir=output_dir,
            verbose=verbose,
        )

        # Run evaluation if patch was generated
        resolved = False
        if agent_result.patch:
            eval_result = await evaluate(
                instance=instance,
                patch=agent_result.patch,
                config=config,
            )
            resolved = eval_result.resolved

        return RunRecord(
            instance_id=instance.instance_id,
            timestamp=datetime.now(),
            success=agent_result.success,
            failure_type=agent_result.failure_type,
            duration_sec=time.perf_counter() - start_time,
            tokens_input=agent_result.tokens_input,
            tokens_output=agent_result.tokens_output,
            tokens_cache_read=agent_result.tokens_cache_read,
            tool_calls_total=agent_result.tool_calls_total,
            tool_calls_by_name=agent_result.tool_calls_by_name,
            cost_usd=agent_result.cost_usd,
            error=agent_result.error,
            patch=agent_result.patch,
            resolved=resolved,
        )

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        return RunRecord(
            instance_id=instance.instance_id,
            timestamp=datetime.now(),
            success=False,
            failure_type=FailureType.UNKNOWN,
            duration_sec=time.perf_counter() - start_time,
            error=f"Git setup failed: {error_msg}",
        )

    except subprocess.TimeoutExpired as e:
        return RunRecord(
            instance_id=instance.instance_id,
            timestamp=datetime.now(),
            success=False,
            failure_type=FailureType.TIMEOUT,
            duration_sec=time.perf_counter() - start_time,
            error=f"Git setup timed out after {e.timeout}s",
        )

    except Exception as e:
        return RunRecord(
            instance_id=instance.instance_id,
            timestamp=datetime.now(),
            success=False,
            failure_type=FailureType.UNKNOWN,
            duration_sec=time.perf_counter() - start_time,
            error=f"Unexpected error: {e}",
        )

    finally:
        # Clean up temp directory
        if work_dir and work_dir.parent.exists():
            shutil.rmtree(work_dir.parent, ignore_errors=True)


async def _prepare_workspace(instance: SWEBenchInstance) -> Path:
    """Clone repo and checkout base commit.

    Args:
        instance: SWE-bench instance with repo info

    Returns:
        Path to workspace directory with repo clone
    """
    base_dir = Path(tempfile.mkdtemp(prefix=f"swe_bench_{instance.instance_id}_"))
    work_dir = base_dir / "repo"

    # Clone repository
    await asyncio.to_thread(
        subprocess.run,
        ["git", "clone", instance.repo_url, str(work_dir)],
        capture_output=True,
        check=True,
        timeout=600,  # 10 minute timeout for clone
    )

    # Checkout base commit
    await asyncio.to_thread(
        subprocess.run,
        ["git", "checkout", instance.base_commit],
        cwd=work_dir,
        capture_output=True,
        check=True,
        timeout=60,
    )

    return work_dir
