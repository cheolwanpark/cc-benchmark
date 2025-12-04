"""Claude agent execution with tool use and token tracking.

This module wraps the Claude Agent SDK to execute agents on SWE-bench instances,
tracking all relevant metrics for benchmarking.
"""

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

from swe_bench_harness.config import BenchmarkConfig, ModelConfig
from swe_bench_harness.dataset import SWEBenchInstance
from swe_bench_harness.metrics import FailureType


@dataclass
class ExecutionResult:
    """Result of executing an agent on a single instance.

    Captures all metrics needed for benchmark analysis.
    """

    success: bool
    failure_type: FailureType
    duration_sec: float
    tokens_input: int
    tokens_output: int
    tokens_cache_read: int
    tool_calls_total: int
    cost_usd: float = 0.0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    error_reason: str | None = None
    patch_generated: str | None = None


# SDK tool names available for benchmarking
SDK_TOOLS = ["Read", "Write", "Bash", "Edit", "Glob", "Grep"]


class ClaudeAgent:
    """Execute Claude agents on SWE-bench instances.

    Handles:
    - Building system and user prompts
    - Agentic execution via claude-agent-sdk
    - Token usage tracking
    - Timeout handling
    - Patch extraction from git diff
    """

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

    def __init__(
        self,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
        resolved_plugins: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            model_config: Model configuration (name)
            benchmark_config: Benchmark configuration (allowed tools, plugins, envs)
            resolved_plugins: Pre-resolved plugin paths (from PluginLoader)
        """
        self.model_config = model_config
        self.benchmark_config = benchmark_config
        self.resolved_plugins = resolved_plugins or []

    async def execute(
        self,
        instance: SWEBenchInstance,
        timeout_sec: int,
        cwd: Path,
    ) -> ExecutionResult:
        """Execute agent on instance with timeout.

        Args:
            instance: SWE-bench instance to solve
            timeout_sec: Maximum execution time in seconds
            cwd: Working directory (temp repo clone)

        Returns:
            ExecutionResult with all metrics
        """
        start_time = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self._execute_inner(instance, cwd),
                timeout=timeout_sec,
            )
            result.duration_sec = time.perf_counter() - start_time
            return result

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                failure_type=FailureType.TIMEOUT,
                duration_sec=time.perf_counter() - start_time,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason=f"Execution timed out after {timeout_sec} seconds",
            )

        except CLINotFoundError:
            return ExecutionResult(
                success=False,
                failure_type=FailureType.API_ERROR,
                duration_sec=time.perf_counter() - start_time,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason="Claude Code CLI not installed",
            )

        except (ProcessError, CLIJSONDecodeError) as e:
            return ExecutionResult(
                success=False,
                failure_type=FailureType.API_ERROR,
                duration_sec=time.perf_counter() - start_time,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason=f"SDK error: {e}",
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                failure_type=FailureType.UNKNOWN,
                duration_sec=time.perf_counter() - start_time,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason=f"Unexpected error: {e}",
            )

    async def _execute_inner(
        self,
        instance: SWEBenchInstance,
        cwd: Path,
    ) -> ExecutionResult:
        """Inner execution logic without timeout handling.

        Args:
            instance: SWE-bench instance to solve
            cwd: Working directory (temp repo clone)

        Returns:
            ExecutionResult with all metrics
        """
        # Validate cwd
        if not cwd.is_dir():
            return ExecutionResult(
                success=False,
                failure_type=FailureType.UNKNOWN,
                duration_sec=0.0,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason=f"Invalid cwd: {cwd}",
            )

        # Build options - don't pass None for env/plugins, let SDK use defaults
        options_kwargs: dict[str, Any] = {
            "allowed_tools": self._resolve_allowed_tools(),
            "permission_mode": "bypassPermissions",
            "system_prompt": self.SYSTEM_PROMPT,
            "model": self.model_config.name,
            "max_turns": 50,
            "cwd": cwd,
        }
        if self.resolved_plugins:
            options_kwargs["plugins"] = self.resolved_plugins
        if self.benchmark_config.envs:
            options_kwargs["env"] = self.benchmark_config.envs

        options = ClaudeAgentOptions(**options_kwargs)

        # Build prompt
        prompt = self._build_user_message(instance)

        # Tracking - accumulate across all messages
        total_input = 0
        total_output = 0
        total_cache_read = 0
        cost_usd = 0.0
        tool_calls_by_name: dict[str, int] = {}
        tool_calls_total = 0

        # Execute agent - collect all messages first to avoid task context issues
        # The SDK uses anyio internally which requires proper task scope handling
        messages = []
        async for message in query(prompt=prompt, options=options):
            messages.append(message)

        # Process collected messages
        for message in messages:
            # Count tool uses from each AssistantMessage
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        tool_calls_total += 1
                        tool_calls_by_name[block.name] = (
                            tool_calls_by_name.get(block.name, 0) + 1
                        )

            # ResultMessage contains final accumulated token counts and cost
            elif isinstance(message, ResultMessage):
                # SDK returns usage as a dict, not an object
                usage = message.usage if message.usage is not None else {}
                total_input = usage.get('input_tokens', 0)
                total_output = usage.get('output_tokens', 0)
                total_cache_read = usage.get('cache_read_input_tokens', 0)

                # total_cost_usd is a direct attribute on ResultMessage
                cost_usd = getattr(message, 'total_cost_usd', None)
                if cost_usd is None:
                    # Fallback: estimate using Claude pricing
                    cost_usd = (
                        (total_input / 1_000_000) * 3.0
                        + (total_output / 1_000_000) * 15.0
                        + (total_cache_read / 1_000_000) * 0.30
                    )

        # Generate patch from git diff (agent modifies files on disk)
        patch = await self._generate_patch_from_git(cwd)
        success = patch is not None and len(patch.strip()) > 0

        return ExecutionResult(
            success=success,
            failure_type=FailureType.NONE if success else FailureType.AGENT_ERROR,
            duration_sec=0.0,  # Will be set by caller
            tokens_input=total_input,
            tokens_output=total_output,
            tokens_cache_read=total_cache_read,
            tool_calls_total=tool_calls_total,
            cost_usd=cost_usd,
            tool_calls_by_name=tool_calls_by_name,
            patch_generated=patch,
            error_reason=None if success else "No changes detected in repo",
        )

    def _build_user_message(self, instance: SWEBenchInstance) -> str:
        """Build the user message from an instance.

        Args:
            instance: SWE-bench instance

        Returns:
            Formatted user message
        """
        return f"""Please fix the following issue in the {instance.repo} repository.

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

    def _resolve_allowed_tools(self) -> list[str]:
        """Resolve allowed tools from benchmark config.

        Returns:
            List of SDK tool names. None in config means all tools.
        """
        allowed = self.benchmark_config.allowed_tools

        # None means all tools
        if allowed is None:
            return SDK_TOOLS.copy()

        # Empty list means no tools
        if not allowed:
            return []

        # Wildcard means all tools
        if "*" in allowed:
            return SDK_TOOLS.copy()

        # Mapping from legacy tool names to SDK tool names
        mapping = {
            "read_file": "Read",
            "write_file": "Write",
            "list_directory": "Bash",  # Use 'ls' via Bash
            "search_code": "Grep",
            "run_command": "Bash",
        }

        result = []
        for tool in allowed:
            if tool in mapping:
                result.append(mapping[tool])
            elif tool in SDK_TOOLS:
                result.append(tool)  # Already SDK format

        return list(set(result))  # Dedupe

    async def _generate_patch_from_git(self, cwd: Path) -> str | None:
        """Generate unified diff patch from git.

        The SDK agent modifies files directly on disk via tools.
        We capture the changes by running git diff, including any new files.

        Args:
            cwd: Working directory (repo root)

        Returns:
            Unified diff string, or None if no changes
        """
        try:
            # Stage all changes including new files so they appear in diff
            await asyncio.to_thread(
                subprocess.run,
                ["git", "add", "-A"],
                cwd=cwd,
                capture_output=True,
                timeout=30,
            )

            # Get diff of staged changes
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "diff", "--staged"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            return None
        except subprocess.TimeoutExpired:
            return None
        except subprocess.SubprocessError:
            return None
        except OSError:
            return None
