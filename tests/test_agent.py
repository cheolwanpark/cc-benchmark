"""Tests for Claude agent execution."""

import pytest

from cc_benchmark.agent import ExecutionResult
from cc_benchmark.metrics import FailureType


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
