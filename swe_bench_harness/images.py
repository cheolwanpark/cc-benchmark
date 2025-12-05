"""Docker image management for Epoch AI SWE-bench images.

This module handles pulling, listing, and managing pre-built Docker images
from the Epoch AI registry (ghcr.io/epoch-research/swe-bench.eval).

Images follow the naming convention:
    ghcr.io/epoch-research/swe-bench.eval.<arch>.<instance_id>

Architectures:
    - x86_64: 2,290 images (full coverage)
    - arm64: 1,819 images (experimental, partial coverage)
"""

from __future__ import annotations

import asyncio
import logging
import platform
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import docker

logger = logging.getLogger(__name__)


# Architecture mapping for normalization
ARCH_MAP = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "AMD64": "x86_64",
    "aarch64": "arm64",  # Linux ARM
    "arm64": "arm64",  # macOS ARM
}


@dataclass(frozen=True)
class ImageInfo:
    """Information about a Docker image.

    Attributes:
        instance_id: SWE-bench instance ID (e.g., 'django__django-16379')
        full_name: Full Docker image name including registry
        architecture: CPU architecture (x86_64 or arm64)
        exists_locally: Whether image is pulled locally
        size_bytes: Image size in bytes (None if not available)
    """

    instance_id: str
    full_name: str
    architecture: str
    exists_locally: bool
    size_bytes: int | None = None

    @property
    def short_name(self) -> str:
        """Get short image name without registry prefix."""
        return self.full_name.split("/")[-1]


class ImageError(Exception):
    """Base exception for image operations."""

    pass


class ImageNotFoundError(ImageError):
    """Image does not exist in registry."""

    pass


class ImagePullError(ImageError):
    """Failed to pull image from registry."""

    pass


class Images:
    """Manage Epoch AI pre-built Docker images for SWE-bench evaluation.

    This class provides methods to pull, list, and manage Docker images from
    the Epoch AI registry. Images are named following the convention:

        ghcr.io/epoch-research/swe-bench.eval.<arch>.<instance_id>

    Example:
        >>> images = Images(registry="ghcr.io/epoch-research/swe-bench.eval")
        >>> images.get("astropy__astropy-13236")
        'ghcr.io/epoch-research/swe-bench.eval.x86_64.astropy__astropy-13236'
        >>> await images.pull(["astropy__astropy-13236"])
        {'astropy__astropy-13236': True}
        >>> images.exists("astropy__astropy-13236")
        True
    """

    # Supported architectures
    ARCH_X86_64 = "x86_64"
    ARCH_ARM64 = "arm64"

    def __init__(
        self,
        registry: str = "ghcr.io/epoch-research/swe-bench.eval",
        architecture: str | None = None,
    ) -> None:
        """Initialize the image manager.

        Args:
            registry: Registry prefix (default: ghcr.io/epoch-research/swe-bench.eval)
            architecture: CPU architecture override (auto-detected if None)
        """
        self._registry = registry.rstrip("/")
        self._arch = architecture or self._detect_architecture()
        self._client: docker.DockerClient | None = None

    @staticmethod
    def _detect_architecture() -> str:
        """Detect CPU architecture for image selection.

        Returns:
            Architecture string (x86_64 or arm64)
        """
        machine = platform.machine()
        arch = ARCH_MAP.get(machine)
        if arch is None:
            logger.warning(f"Unknown architecture '{machine}', defaulting to x86_64")
            return Images.ARCH_X86_64
        return arch

    @property
    def architecture(self) -> str:
        """Get the current architecture setting."""
        return self._arch

    @property
    def registry(self) -> str:
        """Get the registry prefix."""
        return self._registry

    def _get_client(self) -> docker.DockerClient:
        """Lazily initialize and return Docker client.

        Returns:
            Docker client instance

        Raises:
            ImageError: If Docker is not available
        """
        if self._client is None:
            try:
                import docker

                self._client = docker.from_env()
            except ImportError:
                raise ImageError(
                    "docker package not installed. Install with: pip install docker"
                )
            except docker.errors.DockerException as e:
                raise ImageError(f"Docker not available: {e}")
        return self._client

    def get(self, instance_id: str) -> str:
        """Get full image name for an instance.

        Args:
            instance_id: SWE-bench instance ID (e.g., django__django-16379)

        Returns:
            Full image name (e.g., ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-16379)
        """
        return f"{self._registry}.{self._arch}.{instance_id}"

    def get_swebench_tag(self, instance_id: str) -> str:
        """Get swebench-expected image tag for an instance.

        swebench's run_instance with is_remote_image=True expects:
        {namespace}/sweb.eval.{arch}.{instance_id_with_1776}:latest

        We use namespace='local' to trigger remote image mode which skips
        building from env images.

        Args:
            instance_id: SWE-bench instance ID

        Returns:
            swebench-compatible image tag (remote format with 'local' namespace)
        """
        # Replace __ with _1776_ to match swebench's remote image naming
        instance_id_escaped = instance_id.replace("__", "_1776_")
        return f"local/sweb.eval.{self._arch}.{instance_id_escaped}:latest"

    async def _ensure_swebench_tag(self, instance_id: str) -> bool:
        """Ensure swebench-compatible tag exists for an image.

        Creates an alias tag that swebench's run_instance can find.

        Args:
            instance_id: SWE-bench instance ID

        Returns:
            True if tag exists or was created successfully
        """
        epoch_image = self.get(instance_id)
        swebench_tag = self.get_swebench_tag(instance_id)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "tag", epoch_image, swebench_tag],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.debug(f"Tagged {instance_id} as {swebench_tag}")
                return True
            else:
                stderr = result.stderr.decode() if result.stderr else ""
                logger.warning(f"Failed to tag {instance_id}: {stderr}")
                return False
        except Exception as e:
            logger.warning(f"Error tagging {instance_id}: {e}")
            return False

    def exists(self, instance_id: str) -> bool:
        """Check if image exists locally.

        Args:
            instance_id: SWE-bench instance ID

        Returns:
            True if image is pulled locally
        """
        try:
            client = self._get_client()
            image_name = self.get(instance_id)
            client.images.get(image_name)
            return True
        except ImageError:
            # Docker not available, can't check
            return False
        except Exception:
            return False

    def list(self, instance_ids: list[str] | None = None) -> list[ImageInfo]:
        """List images with local availability status.

        Args:
            instance_ids: Optional filter by instance IDs.
                         If None, lists all local images matching the registry.

        Returns:
            List of ImageInfo objects
        """
        try:
            client = self._get_client()
        except ImageError as e:
            logger.warning(f"Cannot list images: {e}")
            return []

        results: list[ImageInfo] = []

        if instance_ids is None:
            # List all local images matching our registry pattern
            prefix = f"{self._registry}.{self._arch}."
            try:
                all_images = client.images.list()
                for image in all_images:
                    for tag in image.tags:
                        if tag.startswith(prefix):
                            instance_id = tag[len(prefix) :]
                            results.append(
                                ImageInfo(
                                    instance_id=instance_id,
                                    full_name=tag,
                                    architecture=self._arch,
                                    exists_locally=True,
                                    size_bytes=image.attrs.get("Size"),
                                )
                            )
            except Exception as e:
                logger.warning(f"Failed to list images: {e}")
        else:
            # Check specific instance IDs
            for instance_id in instance_ids:
                image_name = self.get(instance_id)
                exists = self.exists(instance_id)
                size = None
                if exists:
                    try:
                        img = client.images.get(image_name)
                        size = img.attrs.get("Size")
                    except Exception:
                        pass
                results.append(
                    ImageInfo(
                        instance_id=instance_id,
                        full_name=image_name,
                        architecture=self._arch,
                        exists_locally=exists,
                        size_bytes=size,
                    )
                )

        return results

    async def pull(
        self,
        instance_ids: list[str],
        max_parallel: int = 4,
        skip_existing: bool = True,
    ) -> dict[str, bool]:
        """Pull images for given instances.

        Uses subprocess for Docker pull to avoid thread safety issues with
        the Docker SDK during concurrent operations.

        Args:
            instance_ids: List of instance IDs to pull
            max_parallel: Maximum concurrent pulls
            skip_existing: Skip images already pulled locally

        Returns:
            Dict mapping instance_id -> success (True/False)
        """
        results: dict[str, bool] = {}
        semaphore = asyncio.Semaphore(max_parallel)

        async def pull_one(instance_id: str) -> tuple[str, bool]:
            if skip_existing and self.exists(instance_id):
                logger.debug(f"Skipping existing image: {instance_id}")
                # Ensure swebench tag exists even for cached images
                await self._ensure_swebench_tag(instance_id)
                return instance_id, True

            async with semaphore:
                image_name = self.get(instance_id)
                logger.info(f"Pulling image: {instance_id}")
                try:
                    # Use subprocess for pull (avoids Docker SDK thread safety issues)
                    result = await asyncio.to_thread(
                        subprocess.run,
                        ["docker", "pull", image_name],
                        capture_output=True,
                        timeout=600,  # 10 minute timeout per image
                    )
                    success = result.returncode == 0
                    if success:
                        logger.info(f"Pulled: {instance_id}")
                        # Tag for swebench compatibility
                        await self._ensure_swebench_tag(instance_id)
                    else:
                        stderr = result.stderr.decode() if result.stderr else ""
                        logger.error(f"Failed to pull {instance_id}: {stderr}")
                    return instance_id, success
                except subprocess.TimeoutExpired:
                    logger.error(f"Timeout pulling {instance_id}")
                    return instance_id, False
                except Exception as e:
                    logger.error(f"Error pulling {instance_id}: {e}")
                    return instance_id, False

        tasks = [pull_one(iid) for iid in instance_ids]
        for coro in asyncio.as_completed(tasks):
            instance_id, success = await coro
            results[instance_id] = success

        return results

    async def remove(
        self,
        instance_ids: list[str],
        force: bool = False,
    ) -> dict[str, bool]:
        """Remove images for given instances.

        Args:
            instance_ids: List of instance IDs to remove
            force: Force removal even if containers are running

        Returns:
            Dict mapping instance_id -> success (True/False)
        """
        results: dict[str, bool] = {}

        for instance_id in instance_ids:
            image_name = self.get(instance_id)
            try:
                # Use subprocess for removal (consistent with pull)
                cmd = ["docker", "rmi", image_name]
                if force:
                    cmd.insert(2, "-f")

                result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    timeout=60,
                )
                success = result.returncode == 0
                if success:
                    logger.info(f"Removed image: {instance_id}")
                else:
                    stderr = result.stderr.decode() if result.stderr else ""
                    # Not an error if image doesn't exist
                    if "No such image" in stderr:
                        logger.debug(f"Image already removed: {instance_id}")
                        success = True
                    else:
                        logger.warning(f"Failed to remove {instance_id}: {stderr}")
                results[instance_id] = success
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout removing {instance_id}")
                results[instance_id] = False
            except Exception as e:
                logger.warning(f"Error removing {instance_id}: {e}")
                results[instance_id] = False

        return results

    def cleanup(self) -> None:
        """Close Docker client connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
