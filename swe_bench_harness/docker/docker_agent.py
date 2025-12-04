#!/usr/bin/env python3
"""Docker container entrypoint for Claude agent execution.

This script runs INSIDE the Docker container and:
1. Isolates stdout for protocol messages only (SDK logs go to stderr)
2. Reads configuration from /config/config.json
3. Executes the Claude agent using claude_agent_sdk
4. Streams NDJSON protocol messages to stdout
5. Generates patch via git diff and writes to /output/patch.diff
6. Outputs final result message with metrics

Usage:
    docker run --rm \
        -v /host/repo:/workspace \
        -v /host/config:/config:ro \
        -v /host/output:/output \
        -e CLAUDE_CODE_OAUTH_TOKEN \
        swe-bench-agent:latest
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# =============================================================================
# STDOUT ISOLATION - Must be done BEFORE any other imports that might print
# =============================================================================

# Save original stdout file descriptor for protocol messages
_original_stdout_fd = os.dup(sys.stdout.fileno())
protocol_stdout = os.fdopen(_original_stdout_fd, "w", buffering=1)  # Line buffered

# Redirect default stdout to stderr (catches SDK logs, print(), etc.)
sys.stdout = sys.stderr

# CRITICAL: Also redirect fd 1 itself to stderr so subprocesses (Claude CLI, git)
# that write directly to fd 1 don't pollute the NDJSON protocol stream.
# Without this, any subprocess writing to stdout would corrupt the protocol.
os.dup2(sys.stderr.fileno(), 1)


def emit(line: str) -> None:
    """Emit a protocol message to the dedicated stdout channel.

    Args:
        line: JSON string to emit (without newline)
    """
    print(line, file=protocol_stdout, flush=True)


# =============================================================================
# Now safe to import other modules
# =============================================================================

from claude_agent_sdk import (
    AssistantMessage,
    CLIJSONDecodeError,
    CLINotFoundError,
    ClaudeAgentOptions,
    ProcessError,
    ResultMessage,
    ToolUseBlock,
    query,
)

# Import protocol types - we inline them here to avoid dependency on host package
# This ensures the container is self-contained

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# Protocol Types (inline copy from protocol.py)
# =============================================================================


class ErrorType(str, Enum):
    CONFIG_MISSING = "config_missing"
    CONFIG_INVALID = "config_invalid"
    CLI_NOT_FOUND = "cli_not_found"
    SDK_ERROR = "sdk_error"
    GIT_ERROR = "git_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class UsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class InstanceConfig(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    FAIL_TO_PASS: str = ""
    PASS_TO_PASS: str = ""


class ModelConfigSchema(BaseModel):
    name: str


class BenchmarkConfigSchema(BaseModel):
    allowed_tools: list[str] = Field(default_factory=list)
    plugins: list[dict[str, Any]] = Field(default_factory=list)
    envs: dict[str, str] = Field(default_factory=dict)


class ExecutionConfigSchema(BaseModel):
    timeout_sec: int = 600
    max_turns: int = 50


class ContainerConfig(BaseModel):
    version: str = "1.0"
    instance: InstanceConfig
    model: ModelConfigSchema
    benchmark: BenchmarkConfigSchema = Field(default_factory=BenchmarkConfigSchema)
    execution: ExecutionConfigSchema = Field(default_factory=ExecutionConfigSchema)


# =============================================================================
# Message Serializer
# =============================================================================


class MessageSerializer:
    """Serialize protocol messages to NDJSON."""

    def __init__(self) -> None:
        self._sequence = 0

    def _next_seq(self) -> int:
        seq = self._sequence
        self._sequence += 1
        return seq

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _emit(self, msg: dict[str, Any]) -> None:
        """Emit a message dict as JSON."""
        emit(json.dumps(msg))

    def init(self, session_id: str | None = None, model: str | None = None) -> None:
        self._emit({
            "type": "init",
            "seq": self._next_seq(),
            "timestamp": self._timestamp(),
            "session_id": session_id,
            "model": model,
        })

    def assistant(
        self,
        tool_calls: list[tuple[str, str]] | None = None,
        text_preview: str | None = None,
    ) -> None:
        self._emit({
            "type": "assistant",
            "seq": self._next_seq(),
            "timestamp": self._timestamp(),
            "tool_calls": [{"name": n, "id": i} for n, i in (tool_calls or [])],
            "text_preview": text_preview[:200] if text_preview else None,
        })

    def result(
        self,
        success: bool,
        usage: UsageInfo | None = None,
        cost_usd: float = 0.0,
        tool_calls_total: int = 0,
        tool_calls_by_name: dict[str, int] | None = None,
        duration_sec: float = 0.0,
        error_reason: str | None = None,
    ) -> None:
        u = usage or UsageInfo()
        self._emit({
            "type": "result",
            "seq": self._next_seq(),
            "timestamp": self._timestamp(),
            "success": success,
            "usage": {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_read_input_tokens": u.cache_read_input_tokens,
                "cache_creation_input_tokens": u.cache_creation_input_tokens,
            },
            "cost_usd": cost_usd,
            "tool_calls_total": tool_calls_total,
            "tool_calls_by_name": tool_calls_by_name or {},
            "duration_sec": round(duration_sec, 2),
            "error_reason": error_reason,
        })

    def error(
        self,
        error_type: ErrorType,
        error_message: str,
        recoverable: bool = False,
        tb: str | None = None,
    ) -> None:
        self._emit({
            "type": "error",
            "seq": self._next_seq(),
            "timestamp": self._timestamp(),
            "error_type": error_type.value,
            "error_message": error_message,
            "recoverable": recoverable,
            "traceback": tb,
        })


# =============================================================================
# Agent Execution
# =============================================================================

# Available SDK tools
SDK_TOOLS = ["Read", "Write", "Bash", "Edit", "Glob", "Grep"]


SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing a bug in a codebase.

You will be given:
1. A problem statement describing the issue
2. Access to tools for reading/writing files and running commands

Your goal is to:
1. Understand the problem
2. Locate the relevant code
3. Implement a fix using the available tools
4. Verify your fix works by running tests
"""


def build_user_message(config: ContainerConfig) -> str:
    """Build the user message from config."""
    return f"""Please fix the following issue in the {config.instance.repo} repository.

## Repository
{config.instance.repo}

## Base Commit
{config.instance.base_commit}

## Problem Statement
{config.instance.problem_statement}

## Failing Tests
The following tests should pass after your fix:
{config.instance.FAIL_TO_PASS}

Please analyze the issue, locate the relevant code, and implement a fix using the available tools.
"""


def resolve_allowed_tools(tools: list[str]) -> list[str]:
    """Resolve tool names to SDK format."""
    if not tools or "*" in tools:
        return SDK_TOOLS.copy()

    mapping = {
        "read_file": "Read",
        "write_file": "Write",
        "list_directory": "Bash",
        "search_code": "Grep",
        "run_command": "Bash",
    }

    result = []
    for tool in tools:
        if tool in mapping:
            result.append(mapping[tool])
        elif tool in SDK_TOOLS:
            result.append(tool)

    return list(set(result)) if result else SDK_TOOLS.copy()


async def generate_patch(workspace: Path, output_dir: Path) -> str | None:
    """Generate patch from git diff and write to output file.

    Returns:
        Patch content, or None if no changes
    """
    try:
        # Stage all changes including new files
        await asyncio.to_thread(
            subprocess.run,
            ["git", "add", "-A"],
            cwd=workspace,
            capture_output=True,
            timeout=30,
        )

        # Get diff of staged changes
        result = await asyncio.to_thread(
            subprocess.run,
            ["git", "diff", "--staged"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and result.stdout.strip():
            patch = result.stdout.strip()
            # Write patch to output file
            patch_file = output_dir / "patch.diff"
            patch_file.write_text(patch)
            return patch

        return None

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


async def run_agent(config: ContainerConfig, serializer: MessageSerializer) -> int:
    """Execute the Claude agent and stream results.

    Args:
        config: Container configuration
        serializer: Message serializer for protocol output

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    workspace = Path("/workspace")
    output_dir = Path("/output")
    start_time = time.perf_counter()

    # Emit init message
    serializer.init(model=config.model.name)

    # Build SDK options
    allowed_tools = resolve_allowed_tools(config.benchmark.allowed_tools)
    options_kwargs: dict[str, Any] = {
        "allowed_tools": allowed_tools,
        "permission_mode": "bypassPermissions",
        "system_prompt": SYSTEM_PROMPT,
        "model": config.model.name,
        "max_turns": config.execution.max_turns,
        "cwd": workspace,
    }

    if config.benchmark.plugins:
        options_kwargs["plugins"] = config.benchmark.plugins
    if config.benchmark.envs:
        options_kwargs["env"] = config.benchmark.envs

    options = ClaudeAgentOptions(**options_kwargs)
    prompt = build_user_message(config)

    # Tracking
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
                # Extract tool calls for tracking
                calls: list[tuple[str, str]] = []
                text_preview: str | None = None

                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        calls.append((block.name, block.id))
                        tool_calls_total += 1
                        tool_calls_by_name[block.name] = (
                            tool_calls_by_name.get(block.name, 0) + 1
                        )
                    elif hasattr(block, "text"):
                        text_preview = block.text

                # Emit assistant message
                serializer.assistant(tool_calls=calls, text_preview=text_preview)

            elif isinstance(message, ResultMessage):
                # Extract final usage stats
                usage = message.usage if message.usage else {}
                total_input = usage.get("input_tokens", 0)
                total_output = usage.get("output_tokens", 0)
                total_cache_read = usage.get("cache_read_input_tokens", 0)

                cost_usd = getattr(message, "total_cost_usd", None)
                if cost_usd is None:
                    # Fallback estimate using Claude pricing
                    cost_usd = (
                        (total_input / 1_000_000) * 3.0
                        + (total_output / 1_000_000) * 15.0
                        + (total_cache_read / 1_000_000) * 0.30
                    )

        # Generate patch
        patch = await generate_patch(workspace, output_dir)
        success = patch is not None and len(patch.strip()) > 0
        duration = time.perf_counter() - start_time

        # Emit result
        serializer.result(
            success=success,
            usage=UsageInfo(
                input_tokens=total_input,
                output_tokens=total_output,
                cache_read_input_tokens=total_cache_read,
            ),
            cost_usd=cost_usd,
            tool_calls_total=tool_calls_total,
            tool_calls_by_name=tool_calls_by_name,
            duration_sec=duration,
            error_reason=None if success else "No changes detected in repo",
        )

        return 0

    except CLINotFoundError as e:
        serializer.error(
            ErrorType.CLI_NOT_FOUND,
            f"Claude Code CLI not installed: {e}",
            recoverable=False,
        )
        return 1

    except (ProcessError, CLIJSONDecodeError) as e:
        serializer.error(
            ErrorType.SDK_ERROR,
            f"SDK error: {e}",
            recoverable=False,
            tb=traceback.format_exc(),
        )
        return 2

    except Exception as e:
        serializer.error(
            ErrorType.UNKNOWN,
            f"Unexpected error: {e}",
            recoverable=False,
            tb=traceback.format_exc(),
        )
        return 99


# =============================================================================
# Main Entrypoint
# =============================================================================


def load_config() -> ContainerConfig | None:
    """Load configuration from /config/config.json.

    Returns:
        Parsed config, or None on error
    """
    config_path = Path("/config/config.json")

    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            data = json.load(f)
        return ContainerConfig.model_validate(data)
    except Exception:
        return None


async def main() -> int:
    """Main entrypoint."""
    serializer = MessageSerializer()

    # Load config
    config = load_config()
    if config is None:
        config_path = Path("/config/config.json")
        if not config_path.exists():
            serializer.error(
                ErrorType.CONFIG_MISSING,
                "Config file not found at /config/config.json",
                recoverable=False,
            )
        else:
            serializer.error(
                ErrorType.CONFIG_INVALID,
                "Failed to parse config file",
                recoverable=False,
                tb=traceback.format_exc(),
            )
        return 1

    # Set up signal handler for graceful shutdown
    def handle_signal(signum: int, frame: Any) -> None:
        serializer.error(
            ErrorType.TIMEOUT,
            f"Received signal {signum}, shutting down",
            recoverable=False,
        )
        sys.exit(124)  # Standard timeout exit code

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Run agent
    return await run_agent(config, serializer)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    # Close protocol stdout cleanly
    protocol_stdout.close()
    sys.exit(exit_code)
