#!/usr/bin/env python3
"""Debug script to run a single SWE-bench instance with full visibility.

This script uses the DockerClaudeAgent to run the agent inside a container,
providing isolation and reproducibility.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
import shutil
from pathlib import Path

from swe_bench_harness.agent import DockerClaudeAgent
from swe_bench_harness.config import BenchmarkConfig, DatasetConfig, ExecutionConfig, ModelConfig
from swe_bench_harness.dataset import DatasetLoader, SWEBenchInstance
from swe_bench_harness.evaluation import Evaluation


async def evaluate_patch(
    instance: SWEBenchInstance, patch: str, model_name: str = "debug"
) -> bool | None:
    """Run SWE-bench evaluation using the Evaluation class.

    Returns None if evaluation unavailable or failed.
    """
    config = ExecutionConfig()
    evaluation = Evaluation(config, model_name=model_name)
    try:
        print("  Running evaluation in Docker (this may take several minutes)...")
        result = await evaluation.evaluate(instance, patch, run_id="debug")
        if not result.success:
            print(f"  Evaluation error: {result.error}")
            return None
        return result.resolved
    except Exception as e:
        print(f"  Evaluation error: {e}")
        return None
    finally:
        evaluation.cleanup()


async def run_single_instance():
    print("=" * 80)
    print("DEBUG: Single SWE-bench Instance Run (Docker Mode)")
    print("=" * 80)

    # Step 1: Load dataset and get first instance
    print("\n[1] Loading dataset...")
    loader = DatasetLoader()
    config = DatasetConfig(name="verified", split=":1")
    instances = loader.load(config)
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

    model_config = ModelConfig(name="claude-sonnet-4-5")
    benchmark_config = BenchmarkConfig(
        name="debug-config",
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    )

    agent = DockerClaudeAgent(
        model_config=model_config,
        benchmark_config=benchmark_config,
        docker_image="swe-bench-agent:latest",
        docker_memory="8g",
        docker_cpus=4,
        stream_output=True,
    )

    print(f"  Working directory: {work_dir}")
    print(f"  Docker image: {agent.docker_image}")
    print(f"  Memory limit: {agent.docker_memory}")
    print(f"  CPU limit: {agent.docker_cpus}")
    print(f"  Allowed tools: {benchmark_config.allowed_tools}")
    print(f"  Model: {model_config.name}")

    # Step 4: Execute agent in Docker container
    print("\n[4] Executing Agent in Docker Container...")
    print("=" * 80)
    print("  (Agent running inside container, streaming NDJSON protocol...)")

    timeout_sec = 900  # 15 minutes

    try:
        exec_result = await agent.execute(
            instance=instance,
            timeout_sec=timeout_sec,
            cwd=work_dir,
        )

        print("\n" + "=" * 80)
        print("\n[5] Execution Result:")
        print(f"  Success: {exec_result.success}")
        print(f"  Failure Type: {exec_result.failure_type.name}")
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

        if exec_result.error_reason:
            print(f"  Error: {exec_result.error_reason}")

    except Exception as e:
        print(f"\n!!! AGENT ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exec_result = None

    # Step 6: Check for patch
    print("\n[6] Checking for patch...")

    patch = exec_result.patch_generated if exec_result else None

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
        resolved = await evaluate_patch(instance, patch, model_name=model_config.name)
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
    print(f"  Success: {exec_result.success if exec_result else 'N/A'}")
    print(f"  Duration: {exec_result.duration_sec:.2f}s" if exec_result else "  Duration: N/A")
    print(f"  Tool calls: {exec_result.tool_calls_total if exec_result else 'N/A'}")
    print(f"  Input tokens: {exec_result.tokens_input if exec_result else 'N/A'}")
    print(f"  Output tokens: {exec_result.tokens_output if exec_result else 'N/A'}")
    print(f"  Cost: ${exec_result.cost_usd:.4f}" if exec_result else "  Cost: N/A")
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
