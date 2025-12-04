"""Claude agent execution with tool use and token tracking.

This module wraps the Anthropic SDK to execute agents on SWE-bench instances,
tracking all relevant metrics for benchmarking.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from swe_bench_harness.config import ModelConfig, PluginConfig
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
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    error_reason: str | None = None
    patch_generated: str | None = None


# Available tool definitions for benchmarking
AVAILABLE_TOOLS: dict[str, dict[str, Any]] = {
    "read_file": {
        "name": "read_file",
        "description": "Read the contents of a file at the given path",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                }
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Write content to a file at the given path",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    "list_directory": {
        "name": "list_directory",
        "description": "List contents of a directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list",
                }
            },
            "required": ["path"],
        },
    },
    "search_code": {
        "name": "search_code",
        "description": "Search for a pattern in the codebase",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (optional)",
                },
            },
            "required": ["pattern"],
        },
    },
    "run_command": {
        "name": "run_command",
        "description": "Run a shell command",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


class ClaudeAgent:
    """Execute Claude agents on SWE-bench instances.

    Handles:
    - Building system and user prompts
    - Agentic loop with tool use
    - Token usage tracking
    - Timeout handling
    - Patch extraction from responses
    """

    SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing a bug in a codebase.

You will be given:
1. A problem statement describing the issue
2. Access to tools for reading/writing files and running commands

Your goal is to:
1. Understand the problem
2. Locate the relevant code
3. Implement a fix
4. Verify your fix works

When you have completed the fix, output the changes as a unified diff patch.
Format your patch as:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context
-old line
+new line
 context
```
"""

    def __init__(self, model_config: ModelConfig, plugin_config: PluginConfig) -> None:
        """Initialize the agent.

        Args:
            model_config: Model configuration (name, max_tokens, temperature)
            plugin_config: Plugin configuration (allowed tools)
        """
        self.client = anthropic.Anthropic()
        self.model_config = model_config
        self.plugin_config = plugin_config

    async def execute(
        self,
        instance: SWEBenchInstance,
        timeout_sec: int,
    ) -> ExecutionResult:
        """Execute agent on instance with timeout.

        Args:
            instance: SWE-bench instance to solve
            timeout_sec: Maximum execution time in seconds

        Returns:
            ExecutionResult with all metrics
        """
        start_time = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self._execute_inner(instance),
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

        except anthropic.APIError as e:
            return ExecutionResult(
                success=False,
                failure_type=FailureType.API_ERROR,
                duration_sec=time.perf_counter() - start_time,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
                error_reason=f"API error: {e}",
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

    async def _execute_inner(self, instance: SWEBenchInstance) -> ExecutionResult:
        """Inner execution logic without timeout handling.

        Args:
            instance: SWE-bench instance to solve

        Returns:
            ExecutionResult with all metrics
        """
        # Build initial messages
        user_message = self._build_user_message(instance)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        # Build tools
        tools = self._build_tools()

        # Tracking variables
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        tool_calls_by_name: dict[str, int] = {}
        tool_calls_total = 0
        final_response_text = ""

        # Agentic loop
        max_iterations = 50  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Make API call (run in thread pool since it's sync)
            response = await asyncio.to_thread(
                self._create_message,
                messages=messages,
                tools=tools,
            )

            # Accumulate token usage
            usage = response.usage
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            if hasattr(usage, "cache_read_input_tokens"):
                total_cache_read_tokens += usage.cache_read_input_tokens or 0

            # Process response content
            assistant_content: list[dict[str, Any]] = []
            tool_use_blocks = []

            for block in response.content:
                if block.type == "text":
                    final_response_text += block.text
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_calls_total += 1
                    tool_name = block.name
                    tool_calls_by_name[tool_name] = tool_calls_by_name.get(tool_name, 0) + 1
                    tool_use_blocks.append(block)
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            # Add assistant message
            messages.append({"role": "assistant", "content": assistant_content})

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Agent finished
                break
            elif response.stop_reason == "max_tokens":
                # Ran out of tokens
                break
            elif response.stop_reason == "tool_use":
                # Process tool calls
                tool_results = await self._execute_tools(tool_use_blocks)
                messages.append({"role": "user", "content": tool_results})
            else:
                # Unknown stop reason
                break

        # Extract patch from final response
        patch = self._extract_patch(final_response_text)

        # Determine success (for now, success = generated a patch)
        success = patch is not None and len(patch.strip()) > 0

        return ExecutionResult(
            success=success,
            failure_type=FailureType.NONE if success else FailureType.AGENT_ERROR,
            duration_sec=0.0,  # Will be set by caller
            tokens_input=total_input_tokens,
            tokens_output=total_output_tokens,
            tokens_cache_read=total_cache_read_tokens,
            tool_calls_total=tool_calls_total,
            tool_calls_by_name=tool_calls_by_name,
            patch_generated=patch,
            error_reason=None if success else "No valid patch generated",
        )

    def _create_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> anthropic.types.Message:
        """Create a message using the Anthropic API.

        Args:
            messages: Conversation messages
            tools: Available tools

        Returns:
            API response message
        """
        kwargs: dict[str, Any] = {
            "model": self.model_config.name,
            "max_tokens": self.model_config.max_tokens,
            "temperature": self.model_config.temperature,
            "system": self.SYSTEM_PROMPT,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        return self.client.messages.create(**kwargs)

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

## Test Command
To verify your fix, run: {instance.test_cmd}

Please analyze the issue, locate the relevant code, implement a fix, and provide your changes as a unified diff patch.
"""

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build tool definitions based on allowed_tools configuration.

        Returns:
            List of tool definitions for the API
        """
        allowed = self.plugin_config.allowed_tools

        if not allowed:
            return []

        if "*" in allowed:
            return list(AVAILABLE_TOOLS.values())

        tools = []
        for tool_name in allowed:
            if tool_name in AVAILABLE_TOOLS:
                tools.append(AVAILABLE_TOOLS[tool_name])

        return tools

    async def _execute_tools(
        self,
        tool_use_blocks: list[Any],
    ) -> list[dict[str, Any]]:
        """Execute tool calls and return results.

        Note: This is a mock implementation for benchmarking.
        In a real deployment, these would actually execute the tools.

        Args:
            tool_use_blocks: List of tool use blocks from the API

        Returns:
            List of tool result content blocks
        """
        results = []
        for block in tool_use_blocks:
            # Mock tool execution - return placeholder results
            # In production, this would actually run the tools
            result_content = self._mock_tool_execution(block.name, block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_content,
            })
        return results

    def _mock_tool_execution(self, tool_name: str, tool_input: dict) -> str:
        """Mock tool execution for benchmarking.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters

        Returns:
            Mock result string
        """
        if tool_name == "read_file":
            return f"[Mock] Contents of {tool_input.get('path', 'unknown')}:\n# File content would appear here"
        elif tool_name == "write_file":
            return f"[Mock] Successfully wrote to {tool_input.get('path', 'unknown')}"
        elif tool_name == "list_directory":
            return f"[Mock] Contents of {tool_input.get('path', '.')}:\nfile1.py\nfile2.py\nsubdir/"
        elif tool_name == "search_code":
            return f"[Mock] Search results for '{tool_input.get('pattern', '')}':\nfile.py:10: matching line"
        elif tool_name == "run_command":
            return f"[Mock] Command output for '{tool_input.get('command', '')}':\nCommand executed successfully"
        else:
            return f"[Mock] Tool {tool_name} executed"

    def _extract_patch(self, response_text: str) -> str | None:
        """Extract unified diff patch from response text.

        Args:
            response_text: Full agent response

        Returns:
            Extracted patch string, or None if not found
        """
        # Try to find diff block in code fence
        diff_pattern = r"```diff\s*(.*?)```"
        matches = re.findall(diff_pattern, response_text, re.DOTALL)
        if matches:
            return matches[-1].strip()  # Return last diff block

        # Try to find raw unified diff
        unified_diff_pattern = r"(---\s+\S+.*?(?=\n(?:---|\Z)))"
        matches = re.findall(unified_diff_pattern, response_text, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # Try to find any patch-like content
        if "---" in response_text and "+++" in response_text:
            lines = response_text.split("\n")
            patch_lines = []
            in_patch = False
            for line in lines:
                if line.startswith("---") or line.startswith("+++"):
                    in_patch = True
                if in_patch:
                    patch_lines.append(line)
                    if line.startswith("@@") or line.startswith(" ") or line.startswith("+") or line.startswith("-"):
                        continue
                    elif not line.strip():
                        continue
                    else:
                        # End of patch
                        break
            if patch_lines:
                return "\n".join(patch_lines)

        return None
