"""SWE-bench dataset loading from HuggingFace.

This module handles loading SWE-bench instances from HuggingFace Hub,
parsing them into typed dataclass instances, and caching for repeated runs.
"""

from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from swe_bench_harness.config import DatasetConfig


@dataclass(frozen=True)
class SWEBenchInstance:
    """A single SWE-bench problem instance.

    Attributes:
        instance_id: Unique identifier (e.g., "django__django-16379")
        repo: Repository name (e.g., "django/django")
        base_commit: Git commit hash for clean state
        problem_statement: Issue description (the problem to solve)
        test_patch: Expected patch for reference (unified diff format)
        test_cmd: Command to run tests for verification
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    test_cmd: str

    @property
    def repo_url(self) -> str:
        """Get the GitHub URL for the repository."""
        return f"https://github.com/{self.repo}"

    @property
    def repo_owner(self) -> str:
        """Get the repository owner/organization."""
        return self.repo.split("/")[0] if "/" in self.repo else self.repo

    @property
    def repo_name(self) -> str:
        """Get the repository name without owner."""
        return self.repo.split("/")[1] if "/" in self.repo else self.repo


class DatasetLoader:
    """Load SWE-bench instances from HuggingFace.

    Handles:
    - Fetching dataset from HuggingFace Hub
    - Parsing into typed SWEBenchInstance objects
    - Local caching for repeated runs
    """

    # Field mappings from HuggingFace dataset to our dataclass
    # SWE-bench uses slightly different field names depending on version
    FIELD_MAPPINGS = {
        "instance_id": ["instance_id"],
        "repo": ["repo"],
        "base_commit": ["base_commit"],
        "problem_statement": ["problem_statement"],
        "test_patch": ["test_patch", "patch"],
        "test_cmd": ["test_cmd", "PASS_TO_PASS", "test_directives"],
    }

    def load(self, config: DatasetConfig) -> list[SWEBenchInstance]:
        """Load SWE-bench instances from HuggingFace.

        Args:
            config: Dataset configuration with source, split, etc.

        Returns:
            List of SWEBenchInstance objects

        Raises:
            ValueError: If dataset cannot be loaded or parsed
        """
        cache_dir = Path(config.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            dataset = load_dataset(
                config.source,
                split=config.split,
                cache_dir=str(cache_dir),
                trust_remote_code=True,  # Required for some SWE-bench variants
            )
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{config.source}': {e}") from e

        instances = []
        for row in dataset:
            try:
                instance = self._map_hf_to_instance(row)
                instances.append(instance)
            except Exception as e:
                # Log warning but continue with other instances
                instance_id = row.get("instance_id", "unknown")
                print(f"Warning: Failed to parse instance '{instance_id}': {e}")

        if not instances:
            raise ValueError(
                f"No valid instances found in dataset '{config.source}' "
                f"with split '{config.split}'"
            )

        return instances

    def _map_hf_to_instance(self, row: dict) -> SWEBenchInstance:
        """Map a HuggingFace dataset row to our dataclass.

        Args:
            row: Dictionary from HuggingFace dataset

        Returns:
            Parsed SWEBenchInstance

        Raises:
            KeyError: If required fields are missing
        """

        def get_field(field_name: str) -> str:
            """Get field value, trying multiple possible names."""
            possible_names = self.FIELD_MAPPINGS.get(field_name, [field_name])
            for name in possible_names:
                if name in row and row[name] is not None:
                    return str(row[name])
            raise KeyError(
                f"Required field '{field_name}' not found. "
                f"Tried: {possible_names}. Available: {list(row.keys())}"
            )

        return SWEBenchInstance(
            instance_id=get_field("instance_id"),
            repo=get_field("repo"),
            base_commit=get_field("base_commit"),
            problem_statement=get_field("problem_statement"),
            test_patch=get_field("test_patch"),
            test_cmd=get_field("test_cmd"),
        )

    def get_dataset_info(self, config: DatasetConfig) -> dict:
        """Get metadata about the dataset without loading all instances.

        Args:
            config: Dataset configuration

        Returns:
            Dictionary with dataset metadata
        """
        try:
            from datasets import get_dataset_config_names, get_dataset_infos

            config_names = get_dataset_config_names(config.source)
            infos = get_dataset_infos(config.source)

            return {
                "source": config.source,
                "split": config.split,
                "available_configs": config_names,
                "infos": {k: str(v) for k, v in infos.items()},
            }
        except Exception as e:
            return {
                "source": config.source,
                "split": config.split,
                "error": str(e),
            }
