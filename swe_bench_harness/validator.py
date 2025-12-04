"""Pre-flight validation for benchmark configuration.

This module validates that all requirements are met before starting
an expensive benchmark run.
"""

import os
from pathlib import Path

from swe_bench_harness.config import DatasetConfig, ExperimentConfig


class ConfigValidator:
    """Validates experiment configuration before execution.

    Performs pre-flight checks to ensure:
    - API keys are configured
    - Output directory is writable
    - Dataset is accessible
    - Estimated costs are reasonable
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        self._errors: list[str] = []
        self._warnings: list[str] = []

    def validate_api_key(self) -> bool:
        """Check that Anthropic API key is configured.

        Returns:
            True if API key is present, False otherwise
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self._errors.append(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-key'"
            )
            return False
        # Anthropic keys are typically long strings - basic sanity check
        if len(api_key) < 20:
            self._errors.append(
                "ANTHROPIC_API_KEY appears too short to be valid"
            )
            return False
        return True

    def validate_output_dir(self, config: ExperimentConfig) -> bool:
        """Check that output directory is writable.

        Args:
            config: Experiment configuration

        Returns:
            True if output directory exists or can be created, False otherwise
        """
        output_path = config.get_output_path()

        try:
            output_path.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = output_path / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except PermissionError:
            self._errors.append(
                f"Cannot write to output directory: {output_path}. "
                "Check permissions or choose a different directory."
            )
            return False
        except OSError as e:
            self._errors.append(f"Error accessing output directory {output_path}: {e}")
            return False

    def validate_dataset_access(self, config: DatasetConfig) -> bool:
        """Check that the dataset is accessible.

        Note: This performs a lightweight check without downloading the full dataset.

        Args:
            config: Dataset configuration

        Returns:
            True if dataset appears accessible, False otherwise
        """
        try:
            from datasets import get_dataset_config_names

            # Check if dataset exists on HuggingFace
            # This is a lightweight API call that doesn't download data
            get_dataset_config_names(config.source)
            return True
        except ImportError:
            self._errors.append(
                "HuggingFace datasets library not installed. "
                "Install with: pip install datasets"
            )
            return False
        except Exception as e:
            # Dataset might not exist or network issues
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                self._errors.append(
                    f"Dataset '{config.source}' not found on HuggingFace. "
                    "Check the dataset name."
                )
            elif "connection" in error_str or "network" in error_str:
                self._errors.append(
                    f"Cannot connect to HuggingFace to verify dataset: {e}"
                )
            else:
                # Some datasets don't have named configs, which is fine
                # Just log a warning but don't fail
                pass
            return True  # Assume accessible unless clearly not found

    def estimate_cost(self, config: ExperimentConfig) -> float:
        """Estimate total API cost for the benchmark.

        This is a rough estimate based on:
        - Number of instances × runs × configs
        - Estimated tokens per run (based on typical usage)
        - Default Claude pricing (fallback values)

        Args:
            config: Experiment configuration

        Returns:
            Estimated cost in USD
        """
        # Parse split to estimate instance count
        split = config.dataset.split
        instance_count = self._parse_split_count(split)

        num_configs = len(config.configs)
        runs = config.execution.runs
        total_runs = instance_count * num_configs * runs

        # Rough estimate: 5k input tokens + 2k output tokens per run
        # This is conservative; complex tasks may use more
        avg_input_tokens = 5000
        avg_output_tokens = 2000

        total_input_tokens = total_runs * avg_input_tokens
        total_output_tokens = total_runs * avg_output_tokens

        # Default pricing (fallback - actual cost comes from SDK)
        input_cost_per_mtok = 3.0
        output_cost_per_mtok = 15.0

        input_cost = (total_input_tokens / 1_000_000) * input_cost_per_mtok
        output_cost = (total_output_tokens / 1_000_000) * output_cost_per_mtok

        return input_cost + output_cost

    def _parse_split_count(self, split: str) -> int:
        """Parse split string to estimate instance count.

        Handles both old format (test[:20]) and new simplified format (:20).

        Args:
            split: Split string like ':20', '10:20', 'test[:20]', or 'test'

        Returns:
            Estimated number of instances
        """
        # Default counts for SWE-bench splits
        default_counts = {
            "test": 300,
            "validation": 500,
            "train": 2000,
        }

        # Handle new simplified format: ":10", "20:", "10:20"
        if split.startswith(":") or (
            ":" in split and "[" not in split and not split.startswith("test")
        ):
            # Simplified format without base split name
            parts = split.split(":")
            if len(parts) == 2:
                start = int(parts[0]) if parts[0] else 0
                end_str = parts[1]
                if end_str:
                    return int(end_str) - start
                else:
                    # Open-ended slice like "10:"
                    return default_counts["test"] - start
            return default_counts["test"]

        # Legacy format: extract base split name
        base_split = split.split("[")[0].strip()

        # Check for slice notation
        if "[" in split and ":" in split:
            # Extract slice
            slice_part = split.split("[")[1].rstrip("]")
            parts = slice_part.split(":")

            if len(parts) == 2:
                start = int(parts[0]) if parts[0] else 0
                end_str = parts[1]

                if "%" in end_str:
                    # Percentage slice
                    pct = int(end_str.rstrip("%"))
                    base_count = default_counts.get(base_split, 300)
                    return int(base_count * pct / 100) - start
                elif end_str:
                    # Absolute slice
                    return int(end_str) - start
                else:
                    # Open-ended slice like [10:]
                    return default_counts.get(base_split, 300) - start

        return default_counts.get(base_split, 300)

    def run_all_checks(self, config: ExperimentConfig) -> list[str]:
        """Run all validation checks.

        Args:
            config: Experiment configuration to validate

        Returns:
            List of error messages (empty if all checks pass)
        """
        self._errors = []
        self._warnings = []

        self.validate_api_key()
        self.validate_output_dir(config)
        self.validate_dataset_access(config.dataset)

        # Add cost warning if estimate is high (warning, not error)
        estimated_cost = self.estimate_cost(config)
        if estimated_cost > 10.0:
            self._warnings.append(
                f"Estimated cost: ${estimated_cost:.2f}. "
                "Consider reducing instance count or runs."
            )

        return self._errors

    def get_warnings(self) -> list[str]:
        """Get warning messages from validation.

        Returns:
            List of warning messages
        """
        return self._warnings

    def get_summary(self, config: ExperimentConfig) -> dict:
        """Get a summary of validation results and estimates.

        Args:
            config: Experiment configuration

        Returns:
            Dictionary with validation summary
        """
        split = config.dataset.split
        instance_count = self._parse_split_count(split)
        total_runs = (
            instance_count * len(config.configs) * config.execution.runs
        )

        return {
            "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "output_dir": str(config.get_output_path()),
            "dataset_source": config.dataset.source,
            "estimated_instances": instance_count,
            "num_configs": len(config.configs),
            "runs": config.execution.runs,
            "total_runs": total_runs,
            "estimated_cost_usd": self.estimate_cost(config),
            "timeout_sec": config.execution.timeout_sec,
            "max_parallel": config.execution.max_parallel,
        }
