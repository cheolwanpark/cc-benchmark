"""Report generation for benchmark results.

This module exports benchmark results to JSON, YAML, and HTML formats.
"""

import json
from pathlib import Path

import yaml
from jinja2 import Environment, PackageLoader, select_autoescape

from swe_bench_harness.metrics import BenchmarkResults


class Reporter:
    """Generate benchmark reports in multiple formats.

    Supports:
    - JSON: Complete nested structure for programmatic analysis
    - YAML: Human-readable format for version control
    - HTML: Interactive dashboard with Chart.js visualizations
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize the reporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup Jinja2 environment
        self._env = Environment(
            loader=PackageLoader("swe_bench_harness", "templates"),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def generate_all(self, results: BenchmarkResults) -> dict[str, Path]:
        """Generate all report formats.

        Args:
            results: Benchmark results to export

        Returns:
            Dictionary mapping format name to output path
        """
        return {
            "json": self.generate_json(results),
            "yaml": self.generate_yaml(results),
            "html": self.generate_html(results),
        }

    def generate_json(self, results: BenchmarkResults) -> Path:
        """Generate JSON report.

        Args:
            results: Benchmark results

        Returns:
            Path to generated file
        """
        path = self.output_dir / "report.json"

        with open(path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        return path

    def generate_yaml(self, results: BenchmarkResults) -> Path:
        """Generate YAML report (summary only, no raw records).

        Args:
            results: Benchmark results

        Returns:
            Path to generated file
        """
        path = self.output_dir / "report.yaml"

        # Create summary-only version (no raw records for readability)
        data = results.to_dict()
        summary_data = {
            "experiment_name": data["experiment_name"],
            "timestamp": data["timestamp"],
            "total_duration_sec": data["total_duration_sec"],
            "summary": data["summary"],
            "configs": data["configs"],
        }

        with open(path, "w") as f:
            yaml.dump(summary_data, f, default_flow_style=False, sort_keys=False)

        return path

    def generate_html(self, results: BenchmarkResults) -> Path:
        """Generate HTML report with Chart.js visualizations.

        Args:
            results: Benchmark results

        Returns:
            Path to generated file
        """
        path = self.output_dir / "report.html"

        template = self._env.get_template("report.html")
        chart_data = self._prepare_chart_data(results)

        html = template.render(
            experiment_name=results.experiment_name,
            timestamp=results.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            total_duration=f"{results.total_duration_sec / 60:.1f} minutes",
            total_runs=len(results.records),
            successful_runs=sum(1 for r in results.records if r.success),
            total_cost=sum(r.cost_usd for r in results.records),
            summaries=results.summaries,
            chart_data=chart_data,
        )

        with open(path, "w") as f:
            f.write(html)

        return path

    def _prepare_chart_data(self, results: BenchmarkResults) -> dict:
        """Prepare data for Chart.js visualizations.

        Args:
            results: Benchmark results

        Returns:
            Dictionary with chart data
        """
        summaries = results.summaries

        return {
            "labels": [s.config_name for s in summaries],
            "config_ids": [s.config_id for s in summaries],
            "success_rates": [round(s.success_rate * 100, 1) for s in summaries],
            "durations": [round(s.duration_mean, 1) for s in summaries],
            "tokens": [s.tokens_mean for s in summaries],
            "tool_calls": [round(s.tool_calls_mean, 1) for s in summaries],
            "costs": [round(s.cost_mean, 4) for s in summaries],
            "total_costs": [round(s.cost_total, 2) for s in summaries],
        }
