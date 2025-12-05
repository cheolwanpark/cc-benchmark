"""SWE-bench dataset loading from HuggingFace.

This module handles loading SWE-bench instances from HuggingFace Hub,
parsing them into typed dataclass instances, and caching for repeated runs.
"""

from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from cc_benchmark.config import DatasetConfig


@dataclass(frozen=True)
class SWEBenchInstance:
    """A single SWE-bench problem instance.

    Attributes:
        instance_id: Unique identifier (e.g., "django__django-16379")
        repo: Repository name (e.g., "django/django")
        base_commit: Git commit hash for clean state
        problem_statement: Issue description (the problem to solve)
        test_patch: Expected patch for reference (unified diff format)
        version: Version string for the repository
        patch: Gold patch that fixes the issue
        environment_setup_commit: Commit for environment setup
        hints_text: Hints for solving the problem
        created_at: Timestamp when instance was created
        FAIL_TO_PASS: Tests that should fail before fix and pass after
        PASS_TO_PASS: Tests that should pass both before and after fix
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    # Additional fields required by SWE-bench harness
    version: str = ""
    patch: str = ""
    environment_setup_commit: str = ""
    hints_text: str = ""
    created_at: str = ""
    FAIL_TO_PASS: str = ""  # JSON array as string
    PASS_TO_PASS: str = ""  # JSON array as string

    def to_dict(self) -> dict:
        """Convert to dictionary for SWE-bench evaluation.

        Returns:
            Dictionary with all fields needed by SWE-bench harness
        """
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "test_patch": self.test_patch,
            "version": self.version,
            "patch": self.patch,
            "environment_setup_commit": self.environment_setup_commit,
            "hints_text": self.hints_text,
            "created_at": self.created_at,
            "FAIL_TO_PASS": self.FAIL_TO_PASS,
            "PASS_TO_PASS": self.PASS_TO_PASS,
        }

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
        "test_patch": ["test_patch"],
        "version": ["version"],
        "patch": ["patch"],
        "environment_setup_commit": ["environment_setup_commit"],
        "hints_text": ["hints_text"],
        "created_at": ["created_at"],
        "FAIL_TO_PASS": ["FAIL_TO_PASS"],
        "PASS_TO_PASS": ["PASS_TO_PASS"],
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

        # Resolve split format: ":10" -> "test[:10]", "10:20" -> "test[10:20]"
        split = self._resolve_split(config.split)

        try:
            dataset = load_dataset(
                config.source,  # Uses property to resolve name -> HuggingFace path
                split=split,
                cache_dir=str(cache_dir),
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

    def _resolve_split(self, split: str) -> str:
        """Resolve simplified split format to HuggingFace format.

        Converts simplified format:
            ":10"   -> "test[:10]"
            "20:"   -> "test[20:]"
            "10:20" -> "test[10:20]"

        Already full format is passed through:
            "test[:10]" -> "test[:10]"
            "test"      -> "test"

        Args:
            split: Split specification in simplified or full format

        Returns:
            HuggingFace-compatible split string
        """
        # Already in full format (has brackets)
        if "[" in split:
            return split

        # No colon means simple split name (e.g., "test", "validation")
        if ":" not in split:
            return split

        # Has colon but no brackets - check if it's simplified format
        # Simplified format: starts with ":" or is purely numeric slicing
        # NOT simplified: starts with known split names (e.g., "test:10" is invalid)
        known_splits = ("test", "validation", "train")
        if any(split.startswith(name) for name in known_splits):
            # This looks like malformed format (e.g., "test:10" instead of "test[:10]")
            # Try to salvage by adding brackets
            parts = split.split(":", 1)
            if len(parts) == 2:
                return f"{parts[0]}[:{parts[1]}]"
            return split

        # Simplified format - add "test" prefix and brackets
        return f"test[{split}]"

    def _map_hf_to_instance(self, row: dict) -> SWEBenchInstance:
        """Map a HuggingFace dataset row to our dataclass.

        Args:
            row: Dictionary from HuggingFace dataset

        Returns:
            Parsed SWEBenchInstance

        Raises:
            KeyError: If required fields are missing
        """

        def get_field(field_name: str, required: bool = True) -> str:
            """Get field value, trying multiple possible names."""
            possible_names = self.FIELD_MAPPINGS.get(field_name, [field_name])
            for name in possible_names:
                if name in row and row[name] is not None:
                    return str(row[name])
            if required:
                raise KeyError(
                    f"Required field '{field_name}' not found. "
                    f"Tried: {possible_names}. Available: {list(row.keys())}"
                )
            return ""

        return SWEBenchInstance(
            instance_id=get_field("instance_id"),
            repo=get_field("repo"),
            base_commit=get_field("base_commit"),
            problem_statement=get_field("problem_statement"),
            test_patch=get_field("test_patch"),
            version=get_field("version", required=False),
            patch=get_field("patch", required=False),
            environment_setup_commit=get_field("environment_setup_commit", required=False),
            hints_text=get_field("hints_text", required=False),
            created_at=get_field("created_at", required=False),
            FAIL_TO_PASS=get_field("FAIL_TO_PASS", required=False),
            PASS_TO_PASS=get_field("PASS_TO_PASS", required=False),
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
