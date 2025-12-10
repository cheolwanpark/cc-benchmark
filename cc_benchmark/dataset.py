"""SWE-bench dataset loading from HuggingFace."""

from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from cc_benchmark.config import DatasetConfig


@dataclass(frozen=True)
class SWEBenchInstance:
    """A single SWE-bench problem instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    version: str = ""
    patch: str = ""
    environment_setup_commit: str = ""
    hints_text: str = ""
    created_at: str = ""
    FAIL_TO_PASS: str = ""
    PASS_TO_PASS: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for SWE-bench evaluation."""
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


def _resolve_split(split: str) -> str:
    """Resolve simplified split format to HuggingFace format.

    Examples:
        ":10"   -> "test[:10]"
        "10:20" -> "test[10:20]"
        "test[:10]" -> "test[:10]"
    """
    if "[" in split:
        return split
    if ":" not in split:
        return split
    return f"test[{split}]"


def load_instances(config: DatasetConfig) -> list[SWEBenchInstance]:
    """Load SWE-bench instances from HuggingFace.

    Args:
        config: Dataset configuration

    Returns:
        List of SWEBenchInstance objects
    """
    cache_dir = Path(config.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    split = _resolve_split(config.split)

    dataset = load_dataset(
        config.source,
        split=split,
        cache_dir=str(cache_dir),
    )

    instances = []
    for idx, row in enumerate(dataset):
        # Validate required fields
        instance_id = row.get("instance_id", "")
        repo = row.get("repo", "")
        base_commit = row.get("base_commit", "")

        if not instance_id:
            raise ValueError(f"Row {idx}: missing required field 'instance_id'")
        if not repo:
            raise ValueError(f"Instance {instance_id}: missing required field 'repo'")
        if not base_commit:
            raise ValueError(f"Instance {instance_id}: missing required field 'base_commit'")

        instance = SWEBenchInstance(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            problem_statement=row.get("problem_statement", ""),
            test_patch=row.get("test_patch", ""),
            version=row.get("version", ""),
            patch=row.get("patch", ""),
            environment_setup_commit=row.get("environment_setup_commit", ""),
            hints_text=row.get("hints_text", ""),
            created_at=row.get("created_at", ""),
            FAIL_TO_PASS=row.get("FAIL_TO_PASS", ""),
            PASS_TO_PASS=row.get("PASS_TO_PASS", ""),
        )
        instances.append(instance)

    if not instances:
        msg = f"No instances found in dataset '{config.source}' with split '{config.split}'"
        raise ValueError(msg)

    return instances
