"""Claude agent execution in Docker containers.

This module runs Claude agents in isolated Docker containers,
passing configuration via environment variables and reading
results from mounted output directory.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from cc_benchmark.config import Config
from cc_benchmark.dataset import SWEBenchInstance
from cc_benchmark.metrics import FailureType
from cc_benchmark.prompts import build_system_prompt, build_user_message


@dataclass
class AgentResult:
    """Result of executing an agent on a single instance."""

    success: bool
    failure_type: FailureType
    duration_sec: float
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0
    tool_calls_total: int = 0
    cost_usd: float = 0.0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    patch: str | None = None


async def run_agent(
    instance: SWEBenchInstance,
    config: Config,
    work_dir: Path,
    output_dir: Path,
    verbose: bool = True,
) -> AgentResult:
    """Execute agent in Docker container.

    Args:
        instance: SWE-bench instance to solve
        config: Benchmark configuration
        work_dir: Working directory with cloned repo
        output_dir: Directory for output files (mounted into container)
        verbose: Whether to print agent responses to stderr (default: True)

    Returns:
        AgentResult with execution metrics
    """
    start_time = time.perf_counter()
    container_name = f"cc-benchmark-{uuid.uuid4().hex[:12]}"

    # Ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate prompts on host
    system_prompt = build_system_prompt()
    user_message = build_user_message(
        repo=instance.repo,
        problem=instance.problem_statement,
        fail_to_pass=instance.FAIL_TO_PASS,
        base_commit=instance.base_commit,
        hints_text=instance.hints_text,
    )

    # Base64 encode for safe env var passing
    system_prompt_b64 = base64.b64encode(system_prompt.encode("utf-8")).decode("ascii")
    user_message_b64 = base64.b64encode(user_message.encode("utf-8")).decode("ascii")

    # Build docker command with environment variables
    cmd = _build_docker_command(
        container_name=container_name,
        instance=instance,
        config=config,
        work_dir=work_dir,
        output_dir=output_dir,
        system_prompt_b64=system_prompt_b64,
        user_message_b64=user_message_b64,
    )

    try:
        # Run container with timeout
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Accumulate stderr for error reporting while streaming
        stderr_bytes = []

        async def stream_stderr(stream):
            """Stream stderr to terminal and accumulate for error reporting."""
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    stderr_bytes.append(line)
                    # Only print to terminal if verbose mode is enabled
                    if verbose:
                        print(line.decode(errors='replace'), end='', file=sys.stderr, flush=True)
            except Exception:
                # Silently handle streaming errors to avoid breaking execution
                pass

        async def consume_stdout(stream):
            """Consume stdout to prevent pipe blocking (Docker may write status)."""
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
            except Exception:
                # Silently handle streaming errors
                pass

        # Stream both pipes concurrently
        stderr_task = asyncio.create_task(stream_stderr(process.stderr))
        stdout_task = asyncio.create_task(consume_stdout(process.stdout))

        try:
            # Wait for process with timeout
            await asyncio.wait_for(
                process.wait(),
                timeout=config.execution.timeout_sec + 60,
            )
            # Ensure all output consumed
            await asyncio.gather(stderr_task, stdout_task)

            # Reconstruct stderr for error handling
            stderr = b''.join(stderr_bytes)

        except TimeoutError:
            # Cancel streaming tasks and cleanup
            stderr_task.cancel()
            stdout_task.cancel()
            await asyncio.gather(stderr_task, stdout_task, return_exceptions=True)
            # Kill container
            await _cleanup_container(container_name)
            return AgentResult(
                success=False,
                failure_type=FailureType.TIMEOUT,
                duration_sec=time.perf_counter() - start_time,
                error=f"Container timed out after {config.execution.timeout_sec} seconds",
            )

        # Read results from output files, include stderr if no metadata
        result = _read_results(output_dir, start_time)

        # If container failed and no error from metadata, include stderr
        if not result.success and not result.error and stderr:
            stderr_text = stderr.decode(errors="replace").strip()
            if stderr_text:
                # Truncate stderr if too long
                if len(stderr_text) > 2000:
                    stderr_text = stderr_text[-2000:]
                result.error = f"Container stderr: {stderr_text}"

        return result

    except Exception as e:
        return AgentResult(
            success=False,
            failure_type=FailureType.UNKNOWN,
            duration_sec=time.perf_counter() - start_time,
            error=f"Docker execution error: {e}",
        )


def _build_docker_command(
    container_name: str,
    instance: SWEBenchInstance,
    config: Config,
    work_dir: Path,
    output_dir: Path,
    system_prompt_b64: str,
    user_message_b64: str,
) -> list[str]:
    """Build docker run command with volume mounts and env vars."""
    cmd = [
        "docker", "run",
        "--rm",
        "--platform", "linux/amd64",
        f"--name={container_name}",
        f"--memory={config.execution.docker_memory}",
        f"--cpus={config.execution.docker_cpus}",
        f"--stop-timeout={config.execution.timeout_sec}",
        "--network=bridge",
        # Volume mounts
        "-v", f"{work_dir}:/workspace:rw",
        "-v", f"{output_dir}:/output:rw",
        "-w", "/workspace",
        # Environment variables for prompts (new prompt system)
        "-e", f"CC_SYSTEM_PROMPT={system_prompt_b64}",
        "-e", f"CC_USER_MESSAGE={user_message_b64}",
        "-e", f"MAX_TURNS={config.execution.max_turns}",
        # Environment variables for config (legacy fallback)
        "-e", f"PROBLEM={instance.problem_statement}",
        "-e", f"MODEL={config.model}",
        "-e", f"REPO={instance.repo}",
        "-e", f"INSTANCE_ID={instance.instance_id}",
        "-e", f"BASE_COMMIT={instance.base_commit}",
        "-e", f"FAIL_TO_PASS={instance.FAIL_TO_PASS}",
        "-e", f"TIMEOUT={config.execution.timeout_sec}",
        "-e", "PYTHONUNBUFFERED=1",  # Disable Python buffering for real-time output
    ]

    # Pass through OAuth token
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if oauth_token:
        cmd.extend(["-e", f"CLAUDE_CODE_OAUTH_TOKEN={oauth_token}"])

    # Mount plugins directory if plugins are configured
    plugin_paths = config.get_plugin_paths()
    if plugin_paths:
        # Create a plugins directory and mount all plugins
        for plugin_path in plugin_paths:
            if plugin_path.exists():
                plugin_name = plugin_path.name
                cmd.extend(["-v", f"{plugin_path}:/plugins/{plugin_name}:ro"])

    # Add the image name
    cmd.append(config.execution.docker_image)

    return cmd


def _read_results(output_dir: Path, start_time: float) -> AgentResult:
    """Read results from output directory after container exits."""
    duration = time.perf_counter() - start_time

    # Read metadata.json
    metadata_file = output_dir / "metadata.json"
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text())
        except json.JSONDecodeError:
            metadata = {}
    else:
        metadata = {}

    # Read patch.diff
    patch_file = output_dir / "patch.diff"
    patch = None
    if patch_file.exists():
        patch = patch_file.read_text()
        if not patch.strip():
            patch = None

    success = metadata.get("success", False)
    error = metadata.get("error")

    # Determine failure type
    if success:
        failure_type = FailureType.NONE
    elif error and "timeout" in error.lower():
        failure_type = FailureType.TIMEOUT
    elif error and ("cli" in error.lower() or "sdk" in error.lower()):
        failure_type = FailureType.API_ERROR
    else:
        failure_type = FailureType.AGENT_ERROR

    return AgentResult(
        success=success,
        failure_type=failure_type,
        duration_sec=metadata.get("duration_sec", duration),
        tokens_input=metadata.get("tokens_input", 0),
        tokens_output=metadata.get("tokens_output", 0),
        tokens_cache_read=metadata.get("tokens_cache_read", 0),
        tool_calls_total=metadata.get("tool_calls_total", 0),
        cost_usd=metadata.get("cost_usd", 0.0),
        tool_calls_by_name=metadata.get("tool_calls_by_name", {}),
        error=error,
        patch=patch,
    )


async def _cleanup_container(container_name: str) -> None:
    """Force remove container if still running."""
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "kill", container_name],
            capture_output=True,
            timeout=10,
        )
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass
