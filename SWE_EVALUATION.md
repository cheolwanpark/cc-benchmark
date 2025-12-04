# SWE-bench Evaluation Process

## Overview

This document explains how SWE-bench evaluation works and how our benchmark harness integrates with the official evaluation process.

## Dataset Structure

SWE-bench instances contain these key fields:

| Field | Description |
|-------|-------------|
| `instance_id` | Unique identifier (e.g., `astropy__astropy-12907`) |
| `repo` | Repository name (e.g., `astropy/astropy`) |
| `base_commit` | Git commit hash for the buggy state |
| `problem_statement` | Issue description from GitHub |
| `patch` | Gold solution patch (ground truth) |
| `test_patch` | Tests added/modified to verify the fix |
| `FAIL_TO_PASS` | Test cases that should fail before fix, pass after |
| `PASS_TO_PASS` | Test cases that should pass both before and after |
| `version` | Repository version string |

**Note:** There is NO `test_cmd` field in the official dataset. The harness determines test execution from `FAIL_TO_PASS` and `PASS_TO_PASS` fields.

## How Official Evaluation Works

### 1. Test Specification

The SWE-bench harness creates a `TestSpec` from the instance:

```python
from swebench.harness.test_spec.test_spec import make_test_spec

test_spec = make_test_spec(instance_dict)
# test_spec contains:
#   - FAIL_TO_PASS: tests that must go from failing to passing
#   - PASS_TO_PASS: tests that must remain passing
```

### 2. Docker Evaluation

Evaluation runs in Docker containers:

```python
from swebench.harness.run_evaluation import run_instance

result = run_instance(
    test_spec,      # Created from instance
    prediction,     # Contains model_patch
    rm_image,       # Whether to remove Docker image after
    force_rebuild,  # Whether to force rebuild image
    client,         # Docker client
    run_id,         # Identifier for this run
    timeout,        # Timeout in seconds
)
```

### 3. Resolution Criteria

A patch is considered **resolved** if:
1. All `FAIL_TO_PASS` tests now pass (the fix works)
2. All `PASS_TO_PASS` tests still pass (no regressions)

## Our Harness Integration

### Agent Execution

The agent receives:
- Problem statement
- Repository at base_commit
- Tools to read/write/edit files

The agent generates a patch by modifying files in the repository.

### Evaluation Flow

```
Instance → Agent generates patch → Official SWE-bench evaluation (Docker)
                                          ↓
                                   Run FAIL_TO_PASS tests
                                   Run PASS_TO_PASS tests
                                          ↓
                                   resolved = True/False
```

### Metrics

| Metric | Description |
|--------|-------------|
| `patch_rate` | % of instances where agent generated any patch |
| `resolve_rate` | % of instances where patch passed official evaluation |

## Important Notes

1. **No `test_cmd` needed**: The official harness determines how to run tests based on the repository structure and the specific test identifiers in `FAIL_TO_PASS`/`PASS_TO_PASS`.

2. **Different test frameworks**: Repositories may use pytest, unittest, nose, or custom test runners. The harness handles this automatically.

3. **Docker required**: Evaluation requires Docker to run tests in isolated containers matching the repository's environment.

4. **Test isolation**: Only the specific tests in `FAIL_TO_PASS` and `PASS_TO_PASS` are run, not the entire test suite.

## References

- [SWE-bench Dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench)
- [SWE-bench GitHub](https://github.com/princeton-nlp/SWE-bench)
- [SWE-bench Paper](https://arxiv.org/abs/2310.06770)
