#!/usr/bin/env python3
"""Build the SWE-bench agent Docker image.

Evaluation images are pulled from Epoch AI's pre-built registry
(ghcr.io/epoch-research/swe-bench.eval) by the Images class.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def build_agent_image() -> int:
    """Build the SWE-bench agent Docker image."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docker_dir = project_root / "cc_benchmark" / "docker"
    dockerfile = docker_dir / "Dockerfile"

    if not dockerfile.exists():
        print(f"Error: Dockerfile not found at {dockerfile}", file=sys.stderr)
        return 1

    tag = "cc-benchmark-agent:latest"
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


def main() -> int:
    """Build the agent Docker image."""
    return build_agent_image()


if __name__ == "__main__":
    sys.exit(main())
