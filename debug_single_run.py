#!/usr/bin/env python3
"""Debug script to run a single SWE-bench instance with full visibility."""

import asyncio
import subprocess
import tempfile
import shutil
from pathlib import Path

from claude_agent_sdk import (
    ClaudeAgentOptions,
    query,
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    ToolUseBlock,
    ToolResultBlock,
    TextBlock,
)
from swe_bench_harness.dataset import DatasetLoader, SWEBenchInstance
from swe_bench_harness.config import DatasetConfig


async def evaluate_patch(instance: SWEBenchInstance, patch: str) -> bool | None:
    """Run official SWE-bench evaluation. Returns None if evaluation unavailable."""
    try:
        import docker
        from swebench.harness.run_evaluation import run_instance
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as e:
        print(f"  SWE-bench evaluation unavailable: {e}")
        print("  Install with: pip install swebench docker")
        return None

    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:
        print(f"  Docker not available: {e}")
        print("  Start Docker Desktop to enable evaluation")
        return None

    print("  Creating test spec...")
    try:
        test_spec = make_test_spec(instance.to_dict())
    except Exception as e:
        print(f"  Failed to create test spec: {e}")
        return None

    prediction = {
        "instance_id": instance.instance_id,
        "model_name_or_path": "claude-agent-debug",
        "model_patch": patch,
    }

    print("  Running evaluation in Docker (this may take several minutes)...")
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_instance,
                test_spec,
                prediction,
                False,  # rm_image - keep for reuse
                False,  # force_rebuild
                client,
                "debug",  # run_id
                1800,  # timeout (30 min)
            ),
            timeout=1860,  # 31 min outer timeout
        )
        if result is None:
            print("  Evaluation returned no result")
            return None
        resolved = result.get("resolved", False)
        return resolved
    except asyncio.TimeoutError:
        print("  Evaluation timed out after 30 minutes")
        return None
    except Exception as e:
        print(f"  Evaluation error: {e}")
        return None


async def run_single_instance():
    print("=" * 80)
    print("DEBUG: Single SWE-bench Instance Run")
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

    # Step 3: Build agent options
    print("\n[3] Configuring Claude Agent...")

    system_prompt = """You are an expert software engineer tasked with fixing a bug in a codebase.

You will be given:
1. A problem statement describing the issue
2. Access to tools for reading/writing files and running commands

Your goal is to:
1. Understand the problem
2. Locate the relevant code
3. Implement a fix using the available tools
4. Verify your fix works by running tests
"""

    user_prompt = f"""Please fix the following issue in the {instance.repo} repository.

## Repository
{instance.repo}

## Base Commit
{instance.base_commit}

## Problem Statement
{instance.problem_statement}

## Failing Tests
The following tests should pass after your fix:
{instance.FAIL_TO_PASS}

Please analyze the issue, locate the relevant code, and implement a fix using the available tools.
"""

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        system_prompt=system_prompt,
        model="claude-sonnet-4-5",
        max_turns=50,
        cwd=work_dir,
    )

    print(f"  Working directory: {work_dir}")
    print(f"  Allowed tools: {options.allowed_tools}")
    print(f"  Model: {options.model}")
    print(f"  Max turns: {options.max_turns}")

    # Step 4: Execute agent with full message logging
    print("\n[4] Executing Agent...")
    print("=" * 80)

    total_input = 0
    total_output = 0
    tool_calls = 0
    cost_usd = 0.0
    message_count = 0

    try:
        async for message in query(prompt=user_prompt, options=options):
            message_count += 1

            if isinstance(message, SystemMessage):
                print(f"\n--- SystemMessage #{message_count} ---")
                print(f"  (System initialization)")

            elif isinstance(message, AssistantMessage):
                print(f"\n--- AssistantMessage #{message_count} ---")
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text = block.text[:200] + "..." if len(block.text) > 200 else block.text
                        print(f"  [Text]: {text}")
                    elif isinstance(block, ToolUseBlock):
                        tool_calls += 1
                        print(f"  [ToolUse]: {block.name}")
                        if block.name in ["Read", "Glob", "Grep"]:
                            print(f"    Input: {str(block.input)[:100]}...")
                        else:
                            print(f"    Input: {str(block.input)[:200]}...")
                    elif isinstance(block, ToolResultBlock):
                        result_preview = str(block.content)[:150] + "..." if len(str(block.content)) > 150 else str(block.content)
                        print(f"  [ToolResult]: {result_preview}")
                    else:
                        print(f"  [Block {type(block).__name__}]")

            elif isinstance(message, ResultMessage):
                print(f"\n--- ResultMessage #{message_count} ---")
                # SDK returns usage as a dict, not an object
                usage = message.usage if message.usage is not None else {}
                total_input = usage.get('input_tokens', 0)
                total_output = usage.get('output_tokens', 0)
                total_cache_read = usage.get('cache_read_input_tokens', 0)
                cost_usd = getattr(message, 'total_cost_usd', None)
                if cost_usd is None:
                    # Fallback: estimate using Claude pricing
                    cost_usd = (
                        (total_input / 1_000_000) * 3.0
                        + (total_output / 1_000_000) * 15.0
                        + (total_cache_read / 1_000_000) * 0.30
                    )
                print(f"  Input tokens: {total_input}")
                print(f"  Output tokens: {total_output}")
                print(f"  Cache read tokens: {total_cache_read}")
                print(f"  Cost: ${cost_usd:.4f}")

            else:
                print(f"\n--- {type(message).__name__} #{message_count} ---")

    except Exception as e:
        print(f"\n!!! AGENT ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)

    # Step 5: Check for changes
    print("\n[5] Checking for changes...")

    # Stage all changes
    subprocess.run(["git", "add", "-A"], cwd=work_dir, capture_output=True)

    # Get diff
    result = subprocess.run(
        ["git", "diff", "--staged"],
        cwd=work_dir,
        capture_output=True,
        text=True,
    )

    patch = result.stdout.strip() if result.returncode == 0 else None

    if patch:
        print(f"  Patch generated ({len(patch)} chars):")
        print("-" * 40)
        # Show first 1000 chars of patch
        print(patch[:1000])
        if len(patch) > 1000:
            print(f"... ({len(patch) - 1000} more chars)")
        print("-" * 40)
    else:
        print("  NO CHANGES DETECTED!")

    # Step 6: Run SWE-bench evaluation
    print("\n[6] Running SWE-bench Evaluation...")
    resolved = None
    if patch:
        resolved = await evaluate_patch(instance, patch)
        if resolved is True:
            print("  Result: RESOLVED (patch fixes the issue)")
        elif resolved is False:
            print("  Result: NOT RESOLVED (patch does not fix the issue)")
        else:
            print("  Result: UNKNOWN (evaluation could not complete)")
    else:
        print("  Skipped (no patch generated)")

    # Step 7: Summary
    print("\n[7] Summary")
    print("=" * 80)
    print(f"  Instance: {instance.instance_id}")
    print(f"  Messages: {message_count}")
    print(f"  Tool calls: {tool_calls}")
    print(f"  Input tokens: {total_input}")
    print(f"  Output tokens: {total_output}")
    print(f"  Cost: ${cost_usd:.4f}")
    print(f"  Patch generated: {'Yes' if patch else 'No'}")
    print(f"  Resolved: {'Yes' if resolved else 'No' if resolved is False else 'N/A'}")

    # Step 8: Cleanup
    print("\n[8] Cleaning up...")
    shutil.rmtree(base_dir, ignore_errors=True)
    print("  Done!")


if __name__ == "__main__":
    asyncio.run(run_single_instance())
