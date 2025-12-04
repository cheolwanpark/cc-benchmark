"""Tests for Claude agent execution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_bench_harness.agent import AVAILABLE_TOOLS, ClaudeAgent, ExecutionResult
from swe_bench_harness.config import ModelConfig, PluginConfig
from swe_bench_harness.metrics import FailureType


class TestClaudeAgent:
    """Tests for ClaudeAgent."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(
            name="claude-sonnet-4-5",
            max_tokens=4096,
            temperature=0.2,
        )

    @pytest.fixture
    def plugin_config_no_tools(self) -> PluginConfig:
        """Create plugin configuration with no tools."""
        return PluginConfig(
            id="baseline",
            name="Baseline",
            allowed_tools=[],
        )

    @pytest.fixture
    def plugin_config_with_tools(self) -> PluginConfig:
        """Create plugin configuration with tools."""
        return PluginConfig(
            id="with_tools",
            name="With Tools",
            allowed_tools=["read_file", "write_file"],
        )

    @pytest.fixture
    def plugin_config_all_tools(self) -> PluginConfig:
        """Create plugin configuration with all tools."""
        return PluginConfig(
            id="all_tools",
            name="All Tools",
            allowed_tools=["*"],
        )

    def test_build_tools_empty(self, model_config, plugin_config_no_tools):
        """Test that empty allowed_tools returns empty list."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)
        tools = agent._build_tools()
        assert tools == []

    def test_build_tools_specific(self, model_config, plugin_config_with_tools):
        """Test that specific tools are included."""
        agent = ClaudeAgent(model_config, plugin_config_with_tools)
        tools = agent._build_tools()

        assert len(tools) == 2
        tool_names = {t["name"] for t in tools}
        assert tool_names == {"read_file", "write_file"}

    def test_build_tools_wildcard(self, model_config, plugin_config_all_tools):
        """Test that wildcard includes all tools."""
        agent = ClaudeAgent(model_config, plugin_config_all_tools)
        tools = agent._build_tools()

        assert len(tools) == len(AVAILABLE_TOOLS)

    def test_extract_patch_from_diff_block(self, model_config, plugin_config_no_tools):
        """Test extracting patch from diff code block."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)

        response = """
Here's the fix:

```diff
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def hello():
-    print("old")
+    print("new")
```

This should fix the issue.
"""
        patch = agent._extract_patch(response)

        assert patch is not None
        assert "--- a/file.py" in patch
        assert '+    print("new")' in patch

    def test_extract_patch_no_diff(self, model_config, plugin_config_no_tools):
        """Test that None is returned when no patch found."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)

        response = "I couldn't find a solution to this problem."
        patch = agent._extract_patch(response)

        assert patch is None

    def test_build_user_message(self, model_config, plugin_config_no_tools, sample_instance):
        """Test user message construction."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)
        message = agent._build_user_message(sample_instance)

        assert sample_instance.repo in message
        assert sample_instance.problem_statement in message
        assert sample_instance.test_cmd in message

    @pytest.mark.asyncio
    async def test_execute_timeout(self, model_config, plugin_config_no_tools, sample_instance):
        """Test that timeout is handled correctly."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)

        # Mock the inner execution to sleep longer than timeout
        async def slow_execution(*args, **kwargs):
            import asyncio
            await asyncio.sleep(10)
            return ExecutionResult(
                success=True,
                failure_type=FailureType.NONE,
                duration_sec=10,
                tokens_input=0,
                tokens_output=0,
                tokens_cache_read=0,
                tool_calls_total=0,
            )

        with patch.object(agent, "_execute_inner", slow_execution):
            result = await agent.execute(sample_instance, timeout_sec=1)

        assert result.success is False
        assert result.failure_type == FailureType.TIMEOUT
        assert "timed out" in result.error_reason.lower()

    @pytest.mark.asyncio
    async def test_execute_api_error(self, model_config, plugin_config_no_tools, sample_instance):
        """Test that API errors are handled correctly."""
        import anthropic

        agent = ClaudeAgent(model_config, plugin_config_no_tools)

        # Mock the inner execution to raise API error
        async def raise_api_error(*args, **kwargs):
            # Use APIConnectionError which is simpler to construct
            raise anthropic.APIConnectionError(request=MagicMock())

        with patch.object(agent, "_execute_inner", raise_api_error):
            result = await agent.execute(sample_instance, timeout_sec=60)

        assert result.success is False
        assert result.failure_type == FailureType.API_ERROR
        assert "API error" in result.error_reason


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            failure_type=FailureType.NONE,
            duration_sec=10.5,
            tokens_input=1000,
            tokens_output=500,
            tokens_cache_read=0,
            tool_calls_total=3,
            tool_calls_by_name={"read_file": 2, "write_file": 1},
            patch_generated="--- patch ---",
        )

        assert result.success is True
        assert result.error_reason is None
        assert result.patch_generated is not None

    def test_failure_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            success=False,
            failure_type=FailureType.TIMEOUT,
            duration_sec=60.0,
            tokens_input=0,
            tokens_output=0,
            tokens_cache_read=0,
            tool_calls_total=0,
            error_reason="Execution timed out",
        )

        assert result.success is False
        assert result.failure_type == FailureType.TIMEOUT
        assert result.error_reason is not None
