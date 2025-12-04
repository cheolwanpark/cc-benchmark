"""Scripts package for SWE-bench benchmark tools."""

import os

# Set Docker platform for all containers (agent + swebench evaluation)
# Required for Apple Silicon compatibility with linux/amd64 images
os.environ["DOCKER_DEFAULT_PLATFORM"] = "linux/amd64"
