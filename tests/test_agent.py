"""Tests for Claude agent execution."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_bench_harness.agent import SDK_TOOLS, ClaudeAgent, ExecutionResult
from swe_bench_harness.config import ModelConfig, PluginConfig
from swe_bench_harness.metrics import FailureType


class TestClaudeAgent:
    """Tests for ClaudeAgent."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(
            name="claude-sonnet-4-5",
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

    @pytest.fixture
    def plugin_config_sdk_tools(self) -> PluginConfig:
        """Create plugin configuration with SDK tool names."""
        return PluginConfig(
            id="sdk_tools",
            name="SDK Tools",
            allowed_tools=["Read", "Write", "Grep"],
        )

    @pytest.fixture
    def temp_git_repo(self) -> Path:
        """Create a temporary git repository for testing."""
        import subprocess

        temp_dir = Path(tempfile.mkdtemp(prefix="test_repo_"))
        subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=temp_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=temp_dir,
            capture_output=True,
            check=True,
        )
        # Create initial file and commit
        (temp_dir / "test.py").write_text("print('hello')\n")
        subprocess.run(["git", "add", "."], cwd=temp_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=temp_dir,
            capture_output=True,
            check=True,
        )
        yield temp_dir
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_map_tools_empty(self, model_config, plugin_config_no_tools):
        """Test that empty allowed_tools returns empty list."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)
        tools = agent._map_tools(plugin_config_no_tools.allowed_tools)
        assert tools == []

    def test_map_tools_legacy_names(self, model_config, plugin_config_with_tools):
        """Test that legacy tool names are mapped to SDK names."""
        agent = ClaudeAgent(model_config, plugin_config_with_tools)
        tools = agent._map_tools(plugin_config_with_tools.allowed_tools)

        # read_file -> Read, write_file -> Write
        assert set(tools) == {"Read", "Write"}

    def test_map_tools_wildcard(self, model_config, plugin_config_all_tools):
        """Test that wildcard includes all SDK tools."""
        agent = ClaudeAgent(model_config, plugin_config_all_tools)
        tools = agent._map_tools(plugin_config_all_tools.allowed_tools)

        assert set(tools) == set(SDK_TOOLS)

    def test_map_tools_sdk_names(self, model_config, plugin_config_sdk_tools):
        """Test that SDK tool names pass through unchanged."""
        agent = ClaudeAgent(model_config, plugin_config_sdk_tools)
        tools = agent._map_tools(plugin_config_sdk_tools.allowed_tools)

        assert set(tools) == {"Read", "Write", "Grep"}

    def test_map_tools_dedupes(self, model_config):
        """Test that duplicate tools are deduped."""
        config = PluginConfig(
            id="dupe",
            name="Dupe",
            allowed_tools=["run_command", "Bash"],  # Both map to Bash
        )
        agent = ClaudeAgent(model_config, config)
        tools = agent._map_tools(config.allowed_tools)

        assert tools == ["Bash"]

    @pytest.mark.asyncio
    async def test_generate_patch_from_git_with_changes(
        self, model_config, plugin_config_no_tools, temp_git_repo
    ):
        """Test patch generation when there are changes."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)

        # Make a change
        (temp_git_repo / "test.py").write_text("print('world')\n")

        patch = await agent._generate_patch_from_git(temp_git_repo)

        assert patch is not None
        assert "-print('hello')" in patch
        assert "+print('world')" in patch

    @pytest.mark.asyncio
    async def test_generate_patch_from_git_new_file(
        self, model_config, plugin_config_no_tools, temp_git_repo
    ):
        """Test patch generation captures new files."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)

        # Add a new file
        (temp_git_repo / "new_file.py").write_text("print('new')\n")

        patch = await agent._generate_patch_from_git(temp_git_repo)

        assert patch is not None
        assert "new_file.py" in patch
        assert "+print('new')" in patch

    @pytest.mark.asyncio
    async def test_generate_patch_from_git_no_changes(
        self, model_config, plugin_config_no_tools, temp_git_repo
    ):
        """Test patch generation when there are no changes."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)

        patch = await agent._generate_patch_from_git(temp_git_repo)

        assert patch is None

    @pytest.mark.asyncio
    async def test_generate_patch_from_git_invalid_dir(
        self, model_config, plugin_config_no_tools
    ):
        """Test patch generation with invalid directory."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)

        patch = await agent._generate_patch_from_git(Path("/nonexistent/path"))

        assert patch is None

    def test_build_user_message(
        self, model_config, plugin_config_no_tools, sample_instance
    ):
        """Test user message construction."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)
        message = agent._build_user_message(sample_instance)

        assert sample_instance.repo in message
        assert sample_instance.problem_statement in message
        assert sample_instance.test_cmd in message

    @pytest.mark.asyncio
    async def test_execute_timeout(
        self, model_config, plugin_config_no_tools, sample_instance, temp_git_repo
    ):
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
            result = await agent.execute(sample_instance, timeout_sec=1, cwd=temp_git_repo)

        assert result.success is False
        assert result.failure_type == FailureType.TIMEOUT
        assert "timed out" in result.error_reason.lower()

    @pytest.mark.asyncio
    async def test_execute_cli_not_found(
        self, model_config, plugin_config_no_tools, sample_instance, temp_git_repo
    ):
        """Test that CLI not found error is handled correctly."""
        from claude_agent_sdk import CLINotFoundError

        agent = ClaudeAgent(model_config, plugin_config_no_tools)

        # Mock the inner execution to raise CLINotFoundError
        async def raise_cli_error(*args, **kwargs):
            raise CLINotFoundError()

        with patch.object(agent, "_execute_inner", raise_cli_error):
            result = await agent.execute(sample_instance, timeout_sec=60, cwd=temp_git_repo)

        assert result.success is False
        assert result.failure_type == FailureType.API_ERROR
        assert "CLI not installed" in result.error_reason

    @pytest.mark.asyncio
    async def test_execute_process_error(
        self, model_config, plugin_config_no_tools, sample_instance, temp_git_repo
    ):
        """Test that process errors are handled correctly."""
        from claude_agent_sdk import ProcessError

        agent = ClaudeAgent(model_config, plugin_config_no_tools)

        # Mock the inner execution to raise ProcessError
        async def raise_process_error(*args, **kwargs):
            raise ProcessError(exit_code=1, stderr="Test error")

        with patch.object(agent, "_execute_inner", raise_process_error):
            result = await agent.execute(sample_instance, timeout_sec=60, cwd=temp_git_repo)

        assert result.success is False
        assert result.failure_type == FailureType.API_ERROR
        assert "SDK error" in result.error_reason

    @pytest.mark.asyncio
    async def test_execute_invalid_cwd(
        self, model_config, plugin_config_no_tools, sample_instance
    ):
        """Test that invalid cwd is handled correctly."""
        agent = ClaudeAgent(model_config, plugin_config_no_tools)

        # Mock query to avoid actual SDK calls
        with patch("swe_bench_harness.agent.query") as mock_query:
            mock_query.return_value = AsyncMock()
            result = await agent.execute(
                sample_instance, timeout_sec=60, cwd=Path("/nonexistent/path")
            )

        assert result.success is False
        assert result.failure_type == FailureType.UNKNOWN
        assert "Invalid cwd" in result.error_reason


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
            tool_calls_by_name={"Read": 2, "Write": 1},
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
