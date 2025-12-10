"""SWE-bench evaluation."""

from __future__ import annotations

import asyncio
import platform
import time
from dataclasses import dataclass

from cc_benchmark.config import Config
from cc_benchmark.dataset import SWEBenchInstance


@dataclass
class EvaluationResult:
    """Result of a SWE-bench evaluation run."""

    instance_id: str
    resolved: bool
    error: str | None = None
    duration_sec: float = 0.0


async def evaluate(
    instance: SWEBenchInstance,
    patch: str,
    config: Config,
) -> EvaluationResult:
    """Run SWE-bench evaluation on a patch.

    Args:
        instance: SWE-bench instance
        patch: Generated patch (unified diff format)
        config: Benchmark configuration

    Returns:
        EvaluationResult with resolution status
    """
    start_time = time.perf_counter()

    try:
        import docker
        from swebench.harness.run_evaluation import run_instance
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as e:
        return EvaluationResult(
            instance_id=instance.instance_id,
            resolved=False,
            error=f"Missing dependency: {e}",
            duration_sec=time.perf_counter() - start_time,
        )

    client = None
    try:
        client = docker.from_env()

        # Determine architecture
        arch = "x86_64" if platform.machine() in ("x86_64", "AMD64") else "arm64"

        # Create test spec for local evaluation
        # We use namespace=None and manually pull/tag Epoch AI images to match expected format
        test_spec = make_test_spec(
            instance.to_dict(),
            arch=arch,
            namespace=None,
        )

        # Create prediction
        prediction = {
            "instance_id": instance.instance_id,
            "model_name_or_path": config.model,
            "model_patch": patch,
        }

        # Run evaluation
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_instance,
                test_spec,
                prediction,
                not config.execution.image_cache,  # rm_image
                False,  # force_rebuild
                client,
                "benchmark",
                config.execution.eval_timeout,
            ),
            timeout=config.execution.eval_timeout + 60,
        )

        duration = time.perf_counter() - start_time
        resolved = result.get("resolved", False) if result else False

        return EvaluationResult(
            instance_id=instance.instance_id,
            resolved=resolved,
            error=None,
            duration_sec=duration,
        )

    except TimeoutError:
        return EvaluationResult(
            instance_id=instance.instance_id,
            resolved=False,
            error=f"Evaluation timed out after {config.execution.eval_timeout}s",
            duration_sec=time.perf_counter() - start_time,
        )

    except Exception as e:
        return EvaluationResult(
            instance_id=instance.instance_id,
            resolved=False,
            error=f"Evaluation error: {e}",
            duration_sec=time.perf_counter() - start_time,
        )

    finally:
        # Clean up Docker client to prevent resource leaks
        if client:
            client.close()
