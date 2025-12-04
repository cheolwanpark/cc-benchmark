#!/usr/bin/env python3
"""Build Docker images for SWE-bench agent and evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_agent_image() -> int:
    """Build the SWE-bench agent Docker image."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docker_dir = project_root / "swe_bench_harness" / "docker"
    dockerfile = docker_dir / "Dockerfile"

    if not dockerfile.exists():
        print(f"Error: Dockerfile not found at {dockerfile}", file=sys.stderr)
        return 1

    tag = "swe-bench-agent:latest"
    print(f"Building Docker image: {tag}")

    cmd = [
        "docker", "build",
        "--platform", "linux/amd64",
        "-t", tag,
        "-f", str(dockerfile),
        str(docker_dir),
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\nSuccessfully built: {tag}")

    return result.returncode


def build_env_images(
    dataset_name: str = "verified",
    split: str = ":",
    max_workers: int = 4,
) -> int:
    """Build SWE-bench environment images for a dataset."""
    try:
        import docker
        from swebench.harness.docker_build import build_env_images as swe_build_env
        from swe_bench_harness.dataset import DatasetLoader
        from swe_bench_harness.config import DatasetConfig
    except ImportError as e:
        print(f"Error: Required packages not installed: {e}", file=sys.stderr)
        print("Install with: pip install swebench docker", file=sys.stderr)
        return 1

    # Check Docker availability first (fail fast)
    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:
        print(f"Error: Docker not available: {e}", file=sys.stderr)
        print("Start Docker Desktop and try again", file=sys.stderr)
        return 1

    # Load dataset
    print(f"Loading dataset: {dataset_name} (split: {split})")
    loader = DatasetLoader()
    config = DatasetConfig(name=dataset_name, split=split)
    instances = loader.load(config)

    if not instances:
        print("No instances found in dataset", file=sys.stderr)
        return 1

    print(f"Found {len(instances)} instances")

    # Build environment images
    print(f"\nBuilding environment images (this may take several minutes)...")
    print("Images are cached - subsequent runs will be faster.\n")

    try:
        from swebench.harness.test_spec.test_spec import make_test_spec

        # Build test specs with explicit image tags to avoid parameter mismatch
        # in swebench's get_test_specs_from_dataset function
        test_specs = [
            make_test_spec(
                inst.to_dict(),
                namespace=None,
                base_image_tag="latest",
                env_image_tag="latest",
                instance_image_tag="latest",
            )
            for inst in instances
        ]
        successful, failed = swe_build_env(
            client, test_specs, force_rebuild=False, max_workers=max_workers
        )

        if failed:
            print(f"\nFailed to build {len(failed)} environment images:", file=sys.stderr)
            for img in failed[:10]:  # Show first 10
                print(f"  - {img}", file=sys.stderr)
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more", file=sys.stderr)
            return 1

        print(f"\nSuccessfully built {len(successful)} environment images!")
        return 0
    except Exception as e:
        print(f"\nError building environment images: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Build Docker images."""
    parser = argparse.ArgumentParser(description="Build Docker images for SWE-bench")
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Build only the agent Docker image (skip environment images)",
    )
    parser.add_argument(
        "--dataset",
        default="verified",
        help="Dataset name (default: verified)",
    )
    parser.add_argument(
        "--split",
        default=":",
        help="Dataset split (default: ':' for all instances)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel builds for environment images (default: 4)",
    )
    args = parser.parse_args()

    # Build agent image first
    ret = build_agent_image()
    if ret != 0:
        return ret

    # Build environment images unless --agent-only
    if not args.agent_only:
        return build_env_images(args.dataset, args.split, args.max_workers)

    return 0


if __name__ == "__main__":
    sys.exit(main())
