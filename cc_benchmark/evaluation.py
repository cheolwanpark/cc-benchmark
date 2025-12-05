"""SWE-bench evaluation orchestration.

This module handles running SWE-bench evaluations using Epoch AI's pre-built
Docker images and the official swebench harness.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import docker

    from cc_benchmark.config import ExecutionConfig
    from cc_benchmark.dataset import SWEBenchInstance
    from cc_benchmark.images import Images

logger = logging.getLogger(__name__)

# Lock for serializing evaluations when image_cache=False
_non_cache_lock: asyncio.Lock | None = None


def _get_non_cache_lock() -> asyncio.Lock:
    """Get or create the lock for non-cached evaluations."""
    global _non_cache_lock
    if _non_cache_lock is None:
        _non_cache_lock = asyncio.Lock()
    return _non_cache_lock


@dataclass
class EvaluationResult:
    """Result of a SWE-bench evaluation run.

    Attributes:
        instance_id: SWE-bench instance ID
        resolved: Whether the patch resolved the issue
        error: Error message if evaluation failed (None if successful)
        duration_sec: Time taken for evaluation in seconds
    """

    instance_id: str
    resolved: bool
    error: str | None = None
    duration_sec: float = 0.0

    @property
    def success(self) -> bool:
        """Check if evaluation completed without errors."""
        return self.error is None


class EvaluationError(Exception):
    """Error during SWE-bench evaluation."""

    pass


class Evaluation:
    """Run SWE-bench evaluations in Docker containers.

    This class replaces the `_evaluate_patch()` method in runner.py,
    providing a cleaner interface for:
    - Creating test specs from instances
    - Running evaluations in Docker
    - Managing image lifecycle (with optional caching)
    - Handling timeouts and errors

    Example:
        >>> from cc_benchmark.config import ExecutionConfig
        >>> from cc_benchmark.images import Images
        >>>
        >>> evaluation = Evaluation(
        ...     execution_config=ExecutionConfig(),
        ...     model_name="claude-sonnet-4-5",
        ...     images=Images(),
        ... )
        >>> result = await evaluation.evaluate(instance, patch)
        >>> print(f"Resolved: {result.resolved}")
    """

    def __init__(
        self,
        execution_config: ExecutionConfig,
        model_name: str,
        images: Images | None = None,
    ) -> None:
        """Initialize the evaluation runner.

        Args:
            execution_config: Execution settings (timeout, caching, etc.)
            model_name: Model name for prediction metadata
            images: Optional Images instance (creates one if not provided)
        """
        self._config = execution_config
        self._model_name = model_name
        self._images = images
        self._client: docker.DockerClient | None = None

    def _get_images(self) -> Images:
        """Get or create Images instance."""
        if self._images is None:
            from cc_benchmark.images import Images

            self._images = Images(registry=self._config.image_registry)
        return self._images

    def _get_docker_client(self) -> docker.DockerClient:
        """Lazily initialize Docker client.

        Returns:
            docker.DockerClient instance

        Raises:
            EvaluationError: If Docker is not available
        """
        if self._client is None:
            try:
                import docker

                self._client = docker.from_env()
            except ImportError:
                raise EvaluationError(
                    "docker package not installed. Install with: pip install docker"
                )
            except docker.errors.DockerException as e:
                raise EvaluationError(f"Docker not available: {e}")
        return self._client

    async def evaluate(
        self,
        instance: SWEBenchInstance,
        patch: str,
        run_id: str = "benchmark",
    ) -> EvaluationResult:
        """Run SWE-bench evaluation on a patch.

        Uses Docker to run the test suite and verify the patch
        fixes the issue (Fail-to-Pass tests pass, Pass-to-Pass tests still pass).

        When image_cache is False, evaluations are serialized using a lock to
        prevent race conditions where one evaluation removes an image while
        another is using it.

        Args:
            instance: SWE-bench instance
            patch: Generated patch (unified diff format)
            run_id: Identifier for this evaluation run

        Returns:
            EvaluationResult with resolution status
        """
        # Use lock when not caching to prevent race conditions
        if not self._config.image_cache:
            async with _get_non_cache_lock():
                return await self._evaluate_impl(instance, patch, run_id)
        else:
            return await self._evaluate_impl(instance, patch, run_id)

    async def _evaluate_impl(
        self,
        instance: SWEBenchInstance,
        patch: str,
        run_id: str,
    ) -> EvaluationResult:
        """Internal implementation of evaluate."""
        start_time = time.perf_counter()

        # Lazy imports for optional dependencies
        try:
            from swebench.harness.run_evaluation import run_instance
            from swebench.harness.test_spec.test_spec import make_test_spec
        except ImportError as e:
            logger.warning(f"SWE-bench evaluation unavailable: {e}")
            return EvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                error=f"swebench not installed: {e}",
                duration_sec=time.perf_counter() - start_time,
            )

        try:
            client = self._get_docker_client()
            images = self._get_images()

            # Ensure image is available
            if not images.exists(instance.instance_id):
                logger.info(f"Pulling image for {instance.instance_id}...")
                pull_results = await images.pull([instance.instance_id])
                if not pull_results.get(instance.instance_id, False):
                    return EvaluationResult(
                        instance_id=instance.instance_id,
                        resolved=False,
                        error="Failed to pull evaluation image",
                        duration_sec=time.perf_counter() - start_time,
                    )
            else:
                # Image exists but ensure swebench tag alias exists
                await images._ensure_swebench_tag(instance.instance_id)

            # Create test spec from instance dict
            # Pass arch to match Images class, namespace='local' to enable remote image mode
            # which skips trying to build from env images
            test_spec = make_test_spec(
                instance.to_dict(),
                arch=images.architecture,
                namespace="local",
            )

            # Create prediction dict
            prediction = {
                "instance_id": instance.instance_id,
                "model_name_or_path": self._model_name,
                "model_patch": patch,
            }

            # Determine rm_image based on caching config
            rm_image = not self._config.image_cache

            # Run evaluation in thread pool with timeout enforcement
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    run_instance,
                    test_spec,
                    prediction,
                    rm_image,  # rm_image - remove if not caching
                    False,  # force_rebuild - never rebuild pre-built images
                    client,
                    run_id,
                    self._config.eval_timeout,
                ),
                timeout=self._config.eval_timeout + 60,  # Extra buffer for cleanup
            )

            duration = time.perf_counter() - start_time

            # Check if resolved
            if result is None:
                logger.warning(
                    f"run_instance returned None for {instance.instance_id}"
                )
                return EvaluationResult(
                    instance_id=instance.instance_id,
                    resolved=False,
                    error="Evaluation returned no result",
                    duration_sec=duration,
                )

            resolved = result.get("resolved", False)
            if "resolved" not in result:
                logger.warning(
                    f"Missing 'resolved' key in result for {instance.instance_id}: "
                    f"{result.keys()}"
                )

            logger.debug(
                f"Evaluation result for {instance.instance_id}: resolved={resolved}"
            )

            return EvaluationResult(
                instance_id=instance.instance_id,
                resolved=resolved,
                error=None,
                duration_sec=duration,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Evaluation timed out for {instance.instance_id}")
            return EvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                error=f"Evaluation timed out after {self._config.eval_timeout}s",
                duration_sec=time.perf_counter() - start_time,
            )

        except EvaluationError as e:
            # Return error result instead of raising
            logger.warning(f"Evaluation setup error for {instance.instance_id}: {e}")
            return EvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                error=str(e),
                duration_sec=time.perf_counter() - start_time,
            )

        except Exception as e:
            # Check for Docker-specific errors
            try:
                import docker

                if isinstance(e, docker.errors.DockerException):
                    logger.warning(f"Docker error during evaluation: {e}")
                    return EvaluationResult(
                        instance_id=instance.instance_id,
                        resolved=False,
                        error=f"Docker error: {e}",
                        duration_sec=time.perf_counter() - start_time,
                    )
            except ImportError:
                pass

            logger.exception(f"Evaluation failed for {instance.instance_id}")
            return EvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                error=f"Evaluation error: {e}",
                duration_sec=time.perf_counter() - start_time,
            )

    async def evaluate_batch(
        self,
        instances_and_patches: list[tuple[SWEBenchInstance, str]],
        max_parallel: int = 1,
    ) -> list[EvaluationResult]:
        """Evaluate multiple patches.

        When image_cache is False, this forces sequential evaluation to avoid
        race conditions where one evaluation removes an image while another
        is using it.

        Args:
            instances_and_patches: List of (instance, patch) tuples
            max_parallel: Maximum concurrent evaluations (forced to 1 if not caching)

        Returns:
            List of EvaluationResult objects
        """
        # Force sequential execution when not caching to avoid race conditions
        if not self._config.image_cache:
            effective_parallel = 1
            if max_parallel > 1:
                logger.info(
                    "Forcing sequential evaluation (image_cache=False prevents parallelism)"
                )
        else:
            effective_parallel = max_parallel

        semaphore = asyncio.Semaphore(effective_parallel)

        async def eval_one(
            instance: SWEBenchInstance, patch: str
        ) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate(instance, patch)

        tasks = [
            eval_one(instance, patch) for instance, patch in instances_and_patches
        ]

        return await asyncio.gather(*tasks)

    def cleanup(self) -> None:
        """Close Docker client and cleanup resources."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

        if self._images is not None:
            self._images.cleanup()
            self._images = None
