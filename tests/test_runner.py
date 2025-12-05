"""Tests for benchmark runner."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cc_benchmark.agent import ExecutionResult
from cc_benchmark.metrics import FailureType
from cc_benchmark.runner import BenchmarkRunner, ProgressEvent


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    @pytest.fixture
    def runner(self, sample_experiment_config, sample_instance) -> BenchmarkRunner:
        """Create a benchmark runner with sample config."""
        return BenchmarkRunner(
            config=sample_experiment_config,
            instances=[sample_instance],
        )

    def test_total_runs(self, runner):
        """Test total runs calculation."""
        # 2 configs × 1 instance × 2 runs = 4
        assert runner.total_runs == 4

    @pytest.mark.asyncio
    async def test_run_completes(self, runner):
        """Test that run completes all iterations."""
        # Mock agent execution
        mock_result = ExecutionResult(
            success=True,
            failure_type=FailureType.NONE,
            duration_sec=1.0,
            tokens_input=100,
            tokens_output=50,
            tokens_cache_read=0,
            tool_calls_total=0,
        )

        with patch(
            "cc_benchmark.runner.DockerClaudeAgent.execute",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            records = []
            async for record in runner.run():
                records.append(record)

            assert len(records) == runner.total_runs

    @pytest.mark.asyncio
    async def test_progress_callback(self, runner):
        """Test that progress callback is called."""
        mock_result = ExecutionResult(
            success=True,
            failure_type=FailureType.NONE,
            duration_sec=1.0,
            tokens_input=100,
            tokens_output=50,
            tokens_cache_read=0,
            tool_calls_total=0,
        )

        progress_events = []

        def on_progress(event: ProgressEvent):
            progress_events.append(event)

        with patch(
            "cc_benchmark.runner.DockerClaudeAgent.execute",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            async for _ in runner.run(on_progress=on_progress):
                pass

            # Should have one event per run
            assert len(progress_events) == runner.total_runs

            # Check last event
            last = progress_events[-1]
            assert last.completed == runner.total_runs
            assert last.total == runner.total_runs

    def test_save_checkpoint(self, runner):
        """Test checkpoint saving."""
        # Add some records
        from datetime import datetime
        from cc_benchmark.metrics import RunRecord

        runner.records = [
            RunRecord(
                run_id="test_run",
                instance_id="inst1",
                config_id="cfg1",
                timestamp=datetime.now(),
                success=True,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            runner.save_checkpoint(checkpoint_path)

            assert checkpoint_path.exists()

            with open(checkpoint_path) as f:
                data = json.load(f)

            assert data["completed_runs"] == 1
            assert len(data["records"]) == 1

    def test_load_checkpoint(self, runner):
        """Test checkpoint loading."""
        checkpoint_data = {
            "experiment_name": "test",
            "timestamp": "2024-01-01T12:00:00",
            "completed_runs": 1,
            "total_runs": 4,
            "records": [
                {
                    "run_id": "test_run",
                    "instance_id": "inst1",
                    "config_id": "cfg1",
                    "timestamp": "2024-01-01T12:00:00",
                    "success": True,
                    "failure_type": "none",
                    "duration_sec": 10.0,
                    "tokens_input": 100,
                    "tokens_output": 50,
                    "tokens_cache_read": 0,
                    "tool_calls_total": 0,
                    "cost_usd": 0.01,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

            loaded = runner.load_checkpoint(checkpoint_path)

            assert loaded == 1
            assert len(runner.records) == 1
            assert runner.records[0].run_id == "test_run"

    def test_load_checkpoint_missing_file(self, runner):
        """Test loading non-existent checkpoint."""
        loaded = runner.load_checkpoint(Path("/nonexistent/checkpoint.json"))
        assert loaded == 0

    def test_get_results(self, runner):
        """Test results aggregation."""
        from datetime import datetime
        from cc_benchmark.metrics import RunRecord

        runner._start_time = 0  # Reset start time

        runner.records = [
            RunRecord(
                run_id="run1",
                instance_id="inst1",
                config_id="baseline",
                timestamp=datetime.now(),
                success=True,
                duration_sec=10.0,
                tokens_input=100,
                tokens_output=50,
                cost_usd=0.01,
            ),
            RunRecord(
                run_id="run2",
                instance_id="inst1",
                config_id="with_tools",
                timestamp=datetime.now(),
                success=False,
                duration_sec=20.0,
                tokens_input=200,
                tokens_output=100,
                cost_usd=0.02,
            ),
        ]

        results = runner.get_results()

        assert results.experiment_name == "test-experiment"
        assert len(results.records) == 2
        assert len(results.summaries) == 2

    def test_get_completed_run_ids(self, runner):
        """Test extraction of completed run IDs."""
        from datetime import datetime
        from cc_benchmark.metrics import RunRecord

        runner.records = [
            RunRecord(
                run_id="inst1_cfg1_run0_abc123",
                instance_id="inst1",
                config_id="cfg1",
                timestamp=datetime.now(),
                success=True,
            ),
            RunRecord(
                run_id="inst1_cfg1_run1_def456",
                instance_id="inst1",
                config_id="cfg1",
                timestamp=datetime.now(),
                success=True,
            ),
        ]

        completed = runner.get_completed_run_ids()

        assert ("inst1", "cfg1", 0) in completed
        assert ("inst1", "cfg1", 1) in completed
