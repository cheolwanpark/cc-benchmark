"""Claude agent execution with tool use and token tracking.

This module wraps the Claude Agent SDK to execute agents on SWE-bench instances,
tracking all relevant metrics for benchmarking.

Two execution modes are supported:
1. ClaudeAgent - Direct in-process execution (deprecated)
2. DockerClaudeAgent - Isolated Docker container execution (recommended)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
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

logger = logging.getLogger(__name__)


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


class DockerClaudeAgent:
    """Execute Claude agents inside Docker containers.

    This provides isolation and reproducibility by running the agent
    in a controlled container environment. All SDK logs go to stderr;
    only NDJSON protocol messages are sent to stdout.

    The container requires:
    - CLAUDE_CODE_OAUTH_TOKEN environment variable for authentication
    - Pre-built Docker image (default: swe-bench-agent:latest)

    Volume mounts:
    - /workspace: Repository clone (read-write)
    - /config: Configuration JSON (read-only)
    - /output: Results including patch file (read-write)
    """

    DEFAULT_IMAGE = "swe-bench-agent:latest"
    DEFAULT_MEMORY = "8g"
    DEFAULT_CPUS = 4

    def __init__(
        self,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
        resolved_plugins: list[dict[str, Any]] | None = None,
        docker_image: str | None = None,
        docker_memory: str | None = None,
        docker_cpus: int | None = None,
        stream_output: bool = False,
    ) -> None:
        """Initialize the Docker agent.

        Args:
            model_config: Model configuration (name)
            benchmark_config: Benchmark configuration (allowed tools, plugins, envs)
            resolved_plugins: Pre-resolved plugin paths (from PluginLoader)
            docker_image: Docker image name (default: swe-bench-agent:latest)
            docker_memory: Memory limit (default: 8g)
            docker_cpus: CPU limit (default: 4)
            stream_output: If True, print NDJSON messages as they arrive (default: False)
        """
        self.model_config = model_config
        self.benchmark_config = benchmark_config
        self.resolved_plugins = resolved_plugins or []
        self.docker_image = docker_image or self.DEFAULT_IMAGE
        self.docker_memory = docker_memory or self.DEFAULT_MEMORY
        self.docker_cpus = docker_cpus or self.DEFAULT_CPUS
        self.stream_output = stream_output
        # Per-execution container name (set in execute(), used by _cleanup_container())
        self._container_name: str | None = None

    async def execute(
        self,
        instance: SWEBenchInstance,
        timeout_sec: int,
        cwd: Path,
    ) -> ExecutionResult:
        """Execute agent in Docker container with timeout.

        Args:
            instance: SWE-bench instance to solve
            timeout_sec: Maximum execution time in seconds
            cwd: Working directory (temp repo clone)

        Returns:
            ExecutionResult with all metrics
        """
        start_time = time.perf_counter()

        # Set up directories for container mounts
        base_dir = cwd.parent
        config_dir = base_dir / "config"
        output_dir = base_dir / "output"

        config_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        # Write config file for container
        config_file = config_dir / "config.json"
        config_data = self._build_config(instance, timeout_sec)
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Build docker command
        cmd = self._build_docker_command(cwd, config_dir, output_dir, timeout_sec)

        try:
            result = await asyncio.wait_for(
                self._run_container(cmd, output_dir),
                timeout=timeout_sec + 60,  # Extra buffer for container overhead
            )
            result.duration_sec = time.perf_counter() - start_time
            return result

        except asyncio.TimeoutError:
            # Kill container if still running
            await self._cleanup_container()
            return ExecutionResult(
                success=False,
                failure_type=FailureType.TIMEOUT,
                duration_sec=time.perf_counter() - start_time,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason=f"Container timed out after {timeout_sec} seconds",
            )

        except Exception as e:
            logger.exception("Docker execution error")
            return ExecutionResult(
                success=False,
                failure_type=FailureType.UNKNOWN,
                duration_sec=time.perf_counter() - start_time,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason=f"Docker execution error: {e}",
            )

    def _build_config(self, instance: SWEBenchInstance, timeout_sec: int) -> dict:
        """Build configuration dictionary for container.

        Args:
            instance: SWE-bench instance
            timeout_sec: Timeout in seconds

        Returns:
            Configuration dict matching ContainerConfig schema
        """
        return {
            "version": "1.0",
            "instance": {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "base_commit": instance.base_commit,
                "problem_statement": instance.problem_statement,
                "FAIL_TO_PASS": instance.FAIL_TO_PASS,
                "PASS_TO_PASS": instance.PASS_TO_PASS,
            },
            "model": {
                "name": self.model_config.name,
            },
            "benchmark": {
                "allowed_tools": self._resolve_allowed_tools(),
                "plugins": self.resolved_plugins,
                "envs": self.benchmark_config.envs or {},
            },
            "execution": {
                "timeout_sec": timeout_sec,
                "max_turns": 50,
            },
        }

    def _resolve_allowed_tools(self) -> list[str]:
        """Resolve allowed tools from benchmark config.

        Returns:
            List of SDK tool names.
        """
        allowed = self.benchmark_config.allowed_tools

        # None or wildcard means all tools
        if allowed is None or (allowed and "*" in allowed):
            return SDK_TOOLS.copy()

        # Empty list means no tools
        if not allowed:
            return []

        # Map legacy names to SDK names
        mapping = {
            "read_file": "Read",
            "write_file": "Write",
            "list_directory": "Bash",
            "search_code": "Grep",
            "run_command": "Bash",
        }

        result = []
        for tool in allowed:
            if tool in mapping:
                result.append(mapping[tool])
            elif tool in SDK_TOOLS:
                result.append(tool)

        return list(set(result)) if result else SDK_TOOLS.copy()

    def _build_docker_command(
        self,
        workspace: Path,
        config_dir: Path,
        output_dir: Path,
        timeout_sec: int,
    ) -> list[str]:
        """Build the docker run command with all mounts and settings.

        Args:
            workspace: Repository directory to mount
            config_dir: Config directory to mount
            output_dir: Output directory to mount
            timeout_sec: Timeout for --stop-timeout

        Returns:
            Command list for subprocess
        """
        # Generate unique container name using UUID to avoid collisions in parallel execution
        import uuid
        self._container_name = f"swe-bench-{uuid.uuid4().hex[:12]}"

        cmd = [
            "docker", "run",
            # "--rm",  # Remove container after exit
            f"--name={self._container_name}",  # Unique name for cleanup
            f"--memory={self.docker_memory}",
            f"--cpus={self.docker_cpus}",
            f"--stop-timeout={timeout_sec}",
            "--network=bridge",  # Allow API access
            # Volume mounts
            "-v", f"{workspace}:/workspace:rw",
            "-v", f"{config_dir}:/config:ro",
            "-v", f"{output_dir}:/output:rw",
            # Working directory
            "-w", "/workspace",
        ]

        # Pass through OAuth token for Claude Code authentication
        oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
        if oauth_token:
            cmd.extend(["-e", f"CLAUDE_CODE_OAUTH_TOKEN={oauth_token}"])
        else:
            logger.warning("CLAUDE_CODE_OAUTH_TOKEN not set in environment")

        # Add the image name
        cmd.append(self.docker_image)

        return cmd

    async def _run_container(
        self,
        cmd: list[str],
        output_dir: Path,
    ) -> ExecutionResult:
        """Run Docker container and parse NDJSON output.

        Args:
            cmd: Docker command to execute
            output_dir: Directory where patch file will be written

        Returns:
            ExecutionResult with metrics from container output
        """
        from swe_bench_harness.protocol import (
            AggregatedMetrics,
            ErrorPayload,
            ResultPayload,
            parse_message,
            ParseError,
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        metrics = AggregatedMetrics()
        stderr_buffer: list[str] = []  # Rolling buffer of last N stderr lines
        max_stderr_lines = 50

        async def consume_stdout():
            """Read and process NDJSON from stdout with optional streaming output."""
            async for line in process.stdout:
                line_text = line.decode(errors="replace").strip()
                if not line_text:
                    continue
                try:
                    msg = parse_message(line_text)
                    metrics.process(msg)
                    if self.stream_output:
                        # Log message type for visibility
                        logger.info("[%s] %s", msg.type, line_text[:200])
                except ParseError as e:
                    logger.warning("Failed to parse message: %s", e)

        async def consume_stderr():
            """Drain stderr line-by-line to prevent buffer deadlock."""
            async for line in process.stderr:
                line_text = line.decode(errors="replace").strip()
                if not line_text:
                    continue
                logger.debug("Container stderr: %s", line_text)
                # Keep rolling buffer of last N lines (errors usually at end)
                stderr_buffer.append(line_text)
                if len(stderr_buffer) > max_stderr_lines:
                    stderr_buffer.pop(0)

        try:
            # Consume both streams concurrently to avoid deadlock
            await asyncio.gather(consume_stdout(), consume_stderr())
            returncode = await process.wait()
            if returncode != 0:
                logger.warning("Container exited with code %d", returncode)
                if stderr_buffer:
                    logger.warning("Last stderr lines: %s", "\n".join(stderr_buffer[-10:]))

            # Read patch from output file
            patch_file = output_dir / "patch.diff"
            patch = None
            if patch_file.exists():
                patch = patch_file.read_text()
                if not patch.strip():
                    patch = None

            # Determine success - trust container's determination from ResultPayload
            # The container sets success=true only if a non-empty patch was generated
            success = metrics.success
            failure_type = FailureType.NONE if success else (
                FailureType.TIMEOUT if any(
                    e.error_type.value == "timeout" for e in metrics.errors
                ) else (
                    FailureType.API_ERROR if any(
                        e.error_type.value in ("cli_not_found", "sdk_error")
                        for e in metrics.errors
                    ) else FailureType.AGENT_ERROR
                )
            )

            # Build error reason from errors if any
            error_reason = metrics.error_reason
            if not error_reason and metrics.errors:
                error_reason = metrics.errors[0].error_message
            if not error_reason and not success:
                error_reason = "No patch generated"

            return ExecutionResult(
                success=success,
                failure_type=failure_type,
                duration_sec=0.0,  # Set by caller
                tokens_input=metrics.usage.input_tokens,
                tokens_output=metrics.usage.output_tokens,
                tokens_cache_read=metrics.usage.cache_read_input_tokens,
                tool_calls_total=metrics.tool_calls_total,
                cost_usd=metrics.cost_usd,
                tool_calls_by_name=metrics.tool_calls_by_name,
                patch_generated=patch,
                error_reason=error_reason,
            )

        except Exception as e:
            logger.exception("Error processing container output")
            return ExecutionResult(
                success=False,
                failure_type=FailureType.UNKNOWN,
                duration_sec=0.0,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason=f"Error processing container output: {e}",
            )

    async def _cleanup_container(self) -> None:
        """Force remove container if still running."""
        if not self._container_name:
            return  # No container to clean up

        try:
            # Try to kill and remove the container
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "kill", self._container_name],
                capture_output=True,
                timeout=10,
            )
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", self._container_name],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            logger.debug("Container cleanup failed for %s", self._container_name)
        finally:
            self._container_name = None  # Reset for next execution
