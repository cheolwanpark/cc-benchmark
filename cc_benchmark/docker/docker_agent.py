#!/usr/bin/env python3
"""Docker container entrypoint for Claude agent execution.

This script runs INSIDE the Docker container and:
1. Reads configuration from environment variables
2. Executes the Claude agent using claude_agent_sdk
3. Generates patch via git diff and writes to /output/patch.diff
4. Writes metadata (cost, tokens, duration) to /output/metadata.json

Usage:
    docker run --rm \
        -v /host/repo:/workspace \
        -v /host/output:/output \
        -v /host/plugins:/plugins:ro \
        -e CLAUDE_CODE_OAUTH_TOKEN \
        -e PROBLEM="..." \
        -e MODEL="claude-sonnet-4-5" \
        -e REPO="owner/repo" \
        -e INSTANCE_ID="..." \
        -e FAIL_TO_PASS="..." \
        cc-benchmark-agent:latest
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    CLIJSONDecodeError,
    CLINotFoundError,
    ClaudeAgentOptions,
    ProcessError,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    query,
)

# Retry configuration for CLI crashes
MAX_RETRIES = 2  # Number of retries after initial attempt (total 3 attempts)
RETRY_BASE_DELAY = 5.0  # seconds

# Available SDK tools
SDK_TOOLS = ["Read", "Write", "Bash", "Edit", "Glob", "Grep"]

# Tool argument display configuration
MAX_TOOL_ARG_DISPLAY_LENGTH = 200  # Maximum characters to display for tool arguments


def _format_tool_args(args: dict[str, Any]) -> str:
    """Format tool arguments for brief display.

    Uses JSON serialization with fallback to handle nested structures safely.
    """
    if not args:
        return ""

    try:
        # Use JSON with default=str to handle non-serializable objects
        json_str = json.dumps(args, default=str, ensure_ascii=False)
        # Limit total length
        if len(json_str) > MAX_TOOL_ARG_DISPLAY_LENGTH:
            return json_str[:MAX_TOOL_ARG_DISPLAY_LENGTH - 3] + "..."
        return json_str
    except Exception:
        # Fallback to str() if JSON fails
        result = str(args)
        if len(result) > MAX_TOOL_ARG_DISPLAY_LENGTH:
            return result[:MAX_TOOL_ARG_DISPLAY_LENGTH - 3] + "..."
        return result

def load_prompts_from_env() -> tuple[str, str]:
    """Load prompts from base64-encoded environment variables.

    Returns:
        (system_prompt, user_message)
    """
    # Try to load from new env vars
    system_prompt_b64 = os.environ.get("CC_SYSTEM_PROMPT")
    user_message_b64 = os.environ.get("CC_USER_MESSAGE")

    if system_prompt_b64 and user_message_b64:
        try:
            # Decode from base64
            system_prompt = base64.b64decode(system_prompt_b64).decode("utf-8")
            user_message = base64.b64decode(user_message_b64).decode("utf-8")
            return system_prompt, user_message
        except Exception as e:
            # Base64 decoding failed (malformed or truncated)
            # Fall through to legacy prompts
            print(f"Warning: Failed to decode prompts from env vars: {e}", file=sys.stderr)
            print("Falling back to legacy prompt system", file=sys.stderr)

    # Fallback to old prompts for backwards compatibility
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

    # Build old-style user message from env vars
    repo = os.environ.get("REPO", "")
    problem = os.environ.get("PROBLEM", "")
    fail_to_pass = os.environ.get("FAIL_TO_PASS", "")
    user_message = f"""Please fix the following issue in the {repo} repository.

## Repository
{repo}

## Problem Statement
{problem}

## Failing Tests
The following tests should pass after your fix:
{fail_to_pass}

Please analyze the issue, locate the relevant code, and implement a fix using the available tools.
"""

    return system_prompt, user_message


def generate_patch(workspace: Path, output_dir: Path) -> str | None:
    """Generate patch from git diff and write to output file.

    Returns:
        Patch content, or None if no changes
    """
    try:
        # Stage all changes including new files
        subprocess.run(
            ["git", "add", "-A"],
            cwd=workspace,
            capture_output=True,
            timeout=30,
        )

        # Get diff of staged changes
        result = subprocess.run(
            ["git", "diff", "--staged"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and result.stdout.strip():
            patch = result.stdout
            # Write patch to output file
            patch_file = output_dir / "patch.diff"
            patch_file.write_text(patch)
            return patch

        return None

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def reset_workspace(workspace: Path, base_commit: str | None = None) -> bool:
    """Reset workspace to clean state before retry."""
    reset_target = base_commit or "HEAD"
    try:
        subprocess.run(
            ["git", "reset", "HEAD"],
            cwd=workspace,
            capture_output=True,
            timeout=30,
        )
        result = subprocess.run(
            ["git", "reset", "--hard", reset_target],
            cwd=workspace,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False

        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=workspace,
            capture_output=True,
            timeout=30,
        )
        return True

    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def _is_cli_crash(error: Exception) -> bool:
    """Check if an error indicates a CLI crash (SIGSEGV, etc.)."""
    exit_code = getattr(error, "exit_code", None) or getattr(error, "returncode", None)
    if exit_code is not None:
        if exit_code == -11 or exit_code == 139:
            return True

    error_msg = str(error).lower()
    return any(x in error_msg for x in [
        "exit code -11", "exit code: -11", "exit_code=-11",
        "sigsegv", "signal 11", "segmentation fault",
        "returned non-zero exit status 139"
    ])


def write_metadata(
    output_dir: Path,
    success: bool,
    duration_sec: float,
    tokens_input: int = 0,
    tokens_output: int = 0,
    tokens_cache_read: int = 0,
    cost_usd: float = 0.0,
    tool_calls_total: int = 0,
    tool_calls_by_name: dict[str, int] | None = None,
    error: str | None = None,
) -> None:
    """Write metadata.json with execution results."""
    metadata = {
        "success": success,
        "duration_sec": round(duration_sec, 2),
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_cache_read": tokens_cache_read,
        "cost_usd": round(cost_usd, 6),
        "tool_calls_total": tool_calls_total,
        "tool_calls_by_name": tool_calls_by_name or {},
        "error": error,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_plugins() -> list[dict[str, Any]]:
    """Load plugins from /plugins directory if it exists."""
    plugins_dir = Path("/plugins")
    if not plugins_dir.exists():
        return []

    plugins = []
    for path in plugins_dir.iterdir():
        if path.is_dir():
            plugins.append({"type": "local", "path": str(path)})
    return plugins


async def run_agent() -> int:
    """Execute the Claude agent and write results to files.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    workspace = Path("/workspace")
    output_dir = Path("/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    # Load config from environment variables
    model = os.environ.get("MODEL", "claude-sonnet-4-5")
    base_commit = os.environ.get("BASE_COMMIT")

    # Load prompts from environment variables (with fallback to old prompts)
    system_prompt, prompt = load_prompts_from_env()

    # Get max_turns from env with default of 100 (was 50)
    max_turns = int(os.environ.get("MAX_TURNS", "100"))

    # Build SDK options
    plugins = load_plugins()
    options = ClaudeAgentOptions(
        allowed_tools=SDK_TOOLS,
        permission_mode="bypassPermissions",
        system_prompt=system_prompt,
        model=model,
        max_turns=max_turns,
        cwd=workspace,
        plugins=plugins if plugins else None,
    )

    # Retry loop for CLI crashes
    total_attempts = MAX_RETRIES + 1

    for attempt in range(total_attempts):
        attempt_start = time.perf_counter()
        tool_calls_total = 0
        tool_calls_by_name: dict[str, int] = {}
        total_input = 0
        total_output = 0
        total_cache_read = 0
        cost_usd = 0.0

        try:
            # Execute agent
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            tool_calls_total += 1
                            tool_calls_by_name[block.name] = (
                                tool_calls_by_name.get(block.name, 0) + 1
                            )
                            # Print tool usage with brief argument summary
                            args_str = _format_tool_args(block.input)
                            print(f"[Tool] {block.name}({args_str})", file=sys.stderr, flush=True)
                        elif isinstance(block, TextBlock):
                            # Print agent text responses
                            print(block.text, file=sys.stderr, flush=True)
                        elif isinstance(block, ThinkingBlock):
                            # Print agent thinking
                            print(f"[Thinking] {block.thinking}", file=sys.stderr, flush=True)

                elif isinstance(message, ResultMessage):
                    # Accumulate usage across all ResultMessages
                    usage = message.usage if message.usage else {}
                    total_input += usage.get("input_tokens", 0)
                    total_output += usage.get("output_tokens", 0)
                    total_cache_read += usage.get("cache_read_input_tokens", 0)

                    # SDK reports cumulative cost in final message
                    msg_cost = getattr(message, "total_cost_usd", None)
                    if msg_cost is not None:
                        cost_usd = msg_cost

            # Generate patch
            patch = generate_patch(workspace, output_dir)
            has_patch = patch is not None and len(patch.strip()) > 0
            duration = time.perf_counter() - start_time

            # Fallback cost estimate if SDK didn't provide it
            if cost_usd == 0.0 and (total_input > 0 or total_output > 0):
                cost_usd = (
                    (total_input / 1_000_000) * 3.0
                    + (total_output / 1_000_000) * 15.0
                    + (total_cache_read / 1_000_000) * 0.30
                )

            write_metadata(
                output_dir,
                success=has_patch,  # Success = generated a patch
                duration_sec=duration,
                tokens_input=total_input,
                tokens_output=total_output,
                tokens_cache_read=total_cache_read,
                cost_usd=cost_usd,
                tool_calls_total=tool_calls_total,
                tool_calls_by_name=tool_calls_by_name,
                error=None if has_patch else "No changes detected in repo",
            )
            return 0

        except CLINotFoundError as e:
            duration = time.perf_counter() - start_time
            write_metadata(
                output_dir, False, duration,
                error=f"Claude Code CLI not installed: {e}"
            )
            return 1

        except Exception as e:
            # Unified handler for all other exceptions
            is_crash = _is_cli_crash(e)

            # Retry if it's a CLI crash and we have retries left
            if is_crash and attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                print(f"CLI crash (attempt {attempt + 1}/{total_attempts}), retrying in {delay:.1f}s...", file=sys.stderr)

                # Reset workspace before retry
                if not reset_workspace(workspace, base_commit):
                    # Workspace reset failed, cannot retry safely
                    error_msg = f"CLI crash and workspace reset failed after {attempt + 1} attempts: {e}"
                    write_metadata(output_dir, False, duration, error=error_msg)
                    return 2

                await asyncio.sleep(delay)
                continue

            # No retry: determine error type and exit code
            duration = time.perf_counter() - start_time

            if is_crash:
                # CLI crash with no retries left - exit code 2
                error_msg = f"CLI crash after {attempt + 1} attempts: {e}"
                write_metadata(output_dir, False, duration, error=error_msg)
                return 2
            elif isinstance(e, (ProcessError, CLIJSONDecodeError)):
                # SDK error that's not a crash - exit code 2
                error_msg = f"SDK error: {e}"
                write_metadata(output_dir, False, duration, error=error_msg)
                return 2
            else:
                # Truly unexpected error - exit code 99
                error_msg = f"Unexpected error: {e}\n{traceback.format_exc()}"
                write_metadata(output_dir, False, duration, error=error_msg)
                return 99

    # Should not reach here
    duration = time.perf_counter() - start_time
    write_metadata(output_dir, False, duration, error="Exhausted all retry attempts")
    return 99


def main() -> int:
    """Main entrypoint."""
    # Set up signal handler for graceful shutdown
    def handle_signal(signum: int, frame: Any) -> None:
        output_dir = Path("/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_metadata(output_dir, False, 0, error=f"Received signal {signum}, shutting down")
        sys.exit(124)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    return asyncio.run(run_agent())


if __name__ == "__main__":
    sys.exit(main())
