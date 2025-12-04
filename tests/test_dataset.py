"""Tests for dataset loading."""

import pytest

from swe_bench_harness.config import DatasetConfig
from swe_bench_harness.dataset import DatasetLoader, SWEBenchInstance


class TestSWEBenchInstance:
    """Tests for SWEBenchInstance dataclass."""

    def test_properties(self, sample_instance):
        """Test computed properties."""
        assert sample_instance.repo_url == "https://github.com/test/test-repo"
        assert sample_instance.repo_owner == "test"
        assert sample_instance.repo_name == "test-repo"

    def test_frozen(self, sample_instance):
        """Test that instance is immutable."""
        with pytest.raises(AttributeError):
            sample_instance.instance_id = "new_id"


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    def test_map_hf_to_instance(self):
        """Test mapping from HuggingFace row to instance."""
        loader = DatasetLoader()

        row = {
            "instance_id": "django__django-16379",
            "repo": "django/django",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug",
            "test_patch": "--- a/file.py\n+++ b/file.py",
            "FAIL_TO_PASS": '["tests/test_example.py::test_bug"]',
            "PASS_TO_PASS": '["tests/test_example.py::test_other"]',
        }

        instance = loader._map_hf_to_instance(row)

        assert instance.instance_id == "django__django-16379"
        assert instance.repo == "django/django"
        assert instance.base_commit == "abc123"
        assert instance.problem_statement == "Fix the bug"
        assert instance.FAIL_TO_PASS == '["tests/test_example.py::test_bug"]'
        assert instance.PASS_TO_PASS == '["tests/test_example.py::test_other"]'

    def test_map_hf_with_optional_fields(self):
        """Test mapping with optional fields missing."""
        loader = DatasetLoader()

        # FAIL_TO_PASS and PASS_TO_PASS are optional
        row = {
            "instance_id": "test__test-123",
            "repo": "test/repo",
            "base_commit": "def456",
            "problem_statement": "Problem",
            "test_patch": "--- patch ---",
        }

        instance = loader._map_hf_to_instance(row)

        assert instance.test_patch == "--- patch ---"
        assert instance.FAIL_TO_PASS == ""
        assert instance.PASS_TO_PASS == ""

    def test_map_hf_missing_field(self):
        """Test that missing required field raises error."""
        loader = DatasetLoader()

        row = {
            "instance_id": "test__test-123",
            # Missing 'repo' field
            "base_commit": "abc123",
            "problem_statement": "Problem",
            "test_patch": "patch",
        }

        with pytest.raises(KeyError):
            loader._map_hf_to_instance(row)


class TestDatasetLoaderIntegration:
    """Integration tests for DatasetLoader (require network)."""

    @pytest.mark.skip(reason="Requires network access to HuggingFace")
    def test_load_swebench_lite(self):
        """Test loading from SWE-bench Lite."""
        config = DatasetConfig(
            name="lite",
            split=":2",
        )

        loader = DatasetLoader()
        instances = loader.load(config)

        assert len(instances) == 2
        assert all(isinstance(i, SWEBenchInstance) for i in instances)
        assert all(i.instance_id for i in instances)
