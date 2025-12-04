"""Tests for report generation."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from swe_bench_harness.metrics import BenchmarkResults, ConfigSummary, RunRecord
from swe_bench_harness.reporter import Reporter


class TestReporter:
    """Tests for Reporter."""

    @pytest.fixture
    def sample_results(self, sample_run_records) -> BenchmarkResults:
        """Create sample benchmark results."""
        summaries = [
            ConfigSummary(
                config_id="baseline",
                config_name="Baseline",
                total_runs=2,
                success_count=1,
                success_rate=0.5,
                duration_mean=35.0,
                duration_p50=35.0,
                duration_p90=55.0,
                duration_std=25.0,
                tokens_mean=2250,
                tool_calls_mean=0.0,
                cost_mean=0.0165,
                cost_total=0.033,
            ),
            ConfigSummary(
                config_id="with_tools",
                config_name="With Tools",
                total_runs=1,
                success_count=1,
                success_rate=1.0,
                duration_mean=20.0,
                duration_p50=20.0,
                duration_p90=20.0,
                duration_std=0.0,
                tokens_mean=5000,
                tool_calls_mean=5.0,
                cost_mean=0.032,
                cost_total=0.032,
            ),
        ]

        return BenchmarkResults(
            experiment_name="test-experiment",
            timestamp=datetime.now(),
            total_duration_sec=120.0,
            records=sample_run_records,
            summaries=summaries,
        )

    def test_generate_json(self, sample_results):
        """Test JSON report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = Reporter(tmpdir)
            path = reporter.generate_json(sample_results)

            assert path.exists()
            assert path.suffix == ".json"

            with open(path) as f:
                data = json.load(f)

            assert data["experiment_name"] == "test-experiment"
            assert len(data["configs"]) == 2
            assert len(data["records"]) == 3

    def test_generate_yaml(self, sample_results):
        """Test YAML report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = Reporter(tmpdir)
            path = reporter.generate_yaml(sample_results)

            assert path.exists()
            assert path.suffix == ".yaml"

            with open(path) as f:
                data = yaml.safe_load(f)

            assert data["experiment_name"] == "test-experiment"
            assert len(data["configs"]) == 2
            # YAML report excludes raw records for readability
            assert "records" not in data

    def test_generate_html(self, sample_results):
        """Test HTML report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = Reporter(tmpdir)
            path = reporter.generate_html(sample_results)

            assert path.exists()
            assert path.suffix == ".html"

            with open(path) as f:
                html = f.read()

            # Check for required elements
            assert "test-experiment" in html
            assert "Chart.js" in html or "chart.js" in html
            assert "successChart" in html
            assert "Baseline" in html
            assert "With Tools" in html

    def test_generate_all(self, sample_results):
        """Test generating all report formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = Reporter(tmpdir)
            paths = reporter.generate_all(sample_results)

            assert "json" in paths
            assert "yaml" in paths
            assert "html" in paths

            for format_name, path in paths.items():
                assert path.exists(), f"{format_name} report not created"

    def test_output_dir_created(self, sample_results):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "output"
            reporter = Reporter(output_dir)

            assert output_dir.exists()

    def test_prepare_chart_data(self, sample_results):
        """Test chart data preparation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = Reporter(tmpdir)
            chart_data = reporter._prepare_chart_data(sample_results)

            assert "labels" in chart_data
            assert "success_rates" in chart_data
            assert "durations" in chart_data
            assert "tokens" in chart_data
            assert "costs" in chart_data

            assert len(chart_data["labels"]) == 2
            assert chart_data["labels"] == ["Baseline", "With Tools"]
            assert chart_data["success_rates"] == [50.0, 100.0]
