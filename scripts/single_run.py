#!/usr/bin/env python3
"""Debug script to run a single SWE-bench instance with full visibility.

This script uses the run_agent function to execute the agent inside a Docker
container, providing isolation and reproducibility.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
import shutil
import traceback
from pathlib import Path

from cc_benchmark.agent import run_agent
from cc_benchmark.config import Config, DatasetConfig, ExecutionConfig
from cc_benchmark.dataset import load_instances, SWEBenchInstance
from cc_benchmark.evaluation import evaluate


async def evaluate_patch(
    instance: SWEBenchInstance, patch: str, config: Config
) -> bool | None:
    """Run SWE-bench evaluation using the evaluate function.

    Returns None if evaluation unavailable or failed.
    """
    try:
        print("  Running evaluation in Docker (this may take several minutes)...")
        result = await evaluate(instance, patch, config)
        if result.error is not None:
            print(f"  Evaluation error: {result.error}")
            return None
        return result.resolved
    except Exception as e:
        print(f"  Evaluation error: {e}")
        return None


async def run_single_instance():
    print("=" * 80)
    print("DEBUG: Single SWE-bench Instance Run (Docker Mode)")
    print("=" * 80)

    # Step 1: Load dataset and get first instance
    print("\n[1] Loading dataset...")
    dataset_config = DatasetConfig(name="verified", split=":1")
    instances = load_instances(dataset_config)
    instance = instances[0]

    print(f"  Instance ID: {instance.instance_id}")
    print(f"  Repo: {instance.repo}")
    print(f"  Base Commit: {instance.base_commit[:12]}...")
    print(f"\n  Problem Statement:\n{'-' * 40}")
    print(f"  {instance.problem_statement[:500]}...")
    print(f"{'-' * 40}")

    # Step 2: Clone repo and checkout base commit
    print("\n[2] Setting up repository...")
    base_dir = Path(tempfile.mkdtemp(prefix="debug_swe_"))
    work_dir = base_dir / "repo"

    repo_url = f"{instance.repo_url}.git"
    print(f"  Cloning {repo_url}...")

    try:
        # Full clone required - SWE-bench commits can be very old
        result = subprocess.run(
            ["git", "clone", repo_url, str(work_dir)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  ERROR cloning: {result.stderr}")
            return
        print(f"  Clone successful")

        # Checkout base commit
        print(f"  Checking out {instance.base_commit[:12]}...")
        result = subprocess.run(
            ["git", "checkout", instance.base_commit],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"  ERROR checkout: {result.stderr}")
            return
        print(f"  Checkout successful")

    except subprocess.TimeoutExpired:
        print("  ERROR: Git operation timed out")
        return

    # Step 3: Configure Docker agent
    print("\n[3] Configuring Docker Claude Agent...")

    # Create output directory for agent results
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create unified config with execution settings
    config = Config(
        name="debug-run",
        model="claude-sonnet-4-5",
        dataset=dataset_config,
        output_dir=str(output_dir),
        execution=ExecutionConfig(
            docker_image="cc-benchmark-agent:latest",
            docker_memory="8g",
            docker_cpus=4,
            timeout_sec=900,
        )
    )

    print(f"  Working directory: {work_dir}")
    print(f"  Docker image: {config.execution.docker_image}")
    print(f"  Memory limit: {config.execution.docker_memory}")
    print(f"  CPU limit: {config.execution.docker_cpus}")
    print(f"  Model: {config.model}")

    # Step 4: Execute agent in Docker container
    print("\n[4] Executing Agent in Docker Container...")
    print("=" * 80)
    print("  (Agent running inside container...)")

    try:
        exec_result = await run_agent(
            instance=instance,
            config=config,
            work_dir=work_dir,
            output_dir=output_dir,
        )

        print("\n" + "=" * 80)
        print("\n[5] Execution Result:")
        print(f"  Success: {exec_result.success}")
        failure_type_name = exec_result.failure_type.name if exec_result.failure_type else "None"
        print(f"  Failure Type: {failure_type_name}")
        print(f"  Duration: {exec_result.duration_sec:.2f}s")
        print(f"  Input tokens: {exec_result.tokens_input}")
        print(f"  Output tokens: {exec_result.tokens_output}")
        print(f"  Cache read tokens: {exec_result.tokens_cache_read}")
        print(f"  Tool calls: {exec_result.tool_calls_total}")
        print(f"  Cost: ${exec_result.cost_usd:.4f}")

        if exec_result.tool_calls_by_name:
            print(f"  Tool breakdown:")
            for name, count in sorted(exec_result.tool_calls_by_name.items()):
                print(f"    {name}: {count}")

        if exec_result.error is not None:
            print(f"  Error: {exec_result.error}")

    except Exception as e:
        print(f"\n!!! AGENT ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        exec_result = None

    # Step 6: Check for patch
    print("\n[6] Checking for patch...")

    patch = getattr(exec_result, 'patch', None) if exec_result else None

    if patch:
        print(f"  Patch generated ({len(patch)} chars):")
        print("-" * 40)
        # Show first 1000 chars of patch
        print(patch[:1000])
        if len(patch) > 1000:
            print(f"... ({len(patch) - 1000} more chars)")
        print("-" * 40)
    else:
        print("  NO PATCH GENERATED!")

    # Step 7: Run SWE-bench evaluation
    print("\n[7] Running SWE-bench Evaluation...")
    resolved = None
    if patch:
        resolved = await evaluate_patch(instance, patch, config=config)
        if resolved is True:
            print("  Result: RESOLVED (patch fixes the issue)")
        elif resolved is False:
            print("  Result: NOT RESOLVED (patch does not fix the issue)")
        else:
            print("  Result: UNKNOWN (evaluation could not complete)")
    else:
        print("  Skipped (no patch generated)")

    # Step 8: Summary
    print("\n[8] Summary")
    print("=" * 80)
    print(f"  Instance: {instance.instance_id}")
    if exec_result:
        print(f"  Success: {exec_result.success}")
        print(f"  Duration: {exec_result.duration_sec:.2f}s")
        print(f"  Tool calls: {exec_result.tool_calls_total}")
        print(f"  Input tokens: {exec_result.tokens_input}")
        print(f"  Output tokens: {exec_result.tokens_output}")
        print(f"  Cost: ${exec_result.cost_usd:.4f}")
    else:
        print("  Success: N/A")
        print("  Duration: N/A")
        print("  Tool calls: N/A")
        print("  Input tokens: N/A")
        print("  Output tokens: N/A")
        print("  Cost: N/A")
    print(f"  Patch generated: {'Yes' if patch else 'No'}")
    print(f"  Resolved: {'Yes' if resolved else 'No' if resolved is False else 'N/A'}")

    # Step 9: Cleanup
    print("\n[9] Cleaning up...")
    shutil.rmtree(base_dir, ignore_errors=True)
    print("  Done!")


def main():
    """Entry point for the single run script."""
    asyncio.run(run_single_instance())


if __name__ == "__main__":
    main()
