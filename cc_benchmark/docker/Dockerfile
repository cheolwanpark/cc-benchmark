# Docker image for SWE-bench Claude agent execution
#
# This container provides an isolated environment for running Claude agents
# on SWE-bench instances. It includes Python, Node.js, git, and the Claude Code CLI.
#
# Build: docker build -t cc-benchmark-agent:latest .
# Run:   docker run --rm -v /host/repo:/workspace -v /host/config:/config:ro \
#                   -v /host/output:/output -e CLAUDE_CODE_OAUTH_TOKEN \
#                   cc-benchmark-agent:latest

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x (required for Claude Code CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Set up Python environment
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the agent entrypoint
COPY docker_agent.py .

# Create mount point directories
RUN mkdir -p /workspace /config /output

# Create non-root user for running the agent
# Note: UID 1000 is used as a common default. On macOS Docker Desktop,
# file permissions are handled via virtualization layer.
RUN useradd -m -u 1000 -s /bin/bash agent \
    && chown -R agent:agent /app

# Configure git system-wide (applies to all users including agent)
RUN git config --system user.email "cc-benchmark-agent@example.com" \
    && git config --system user.name "CC-Benchmark Agent" \
    && git config --system init.defaultBranch main

# Switch to non-root user
USER agent

# Set working directory to the mounted repo
WORKDIR /workspace

# Run the agent entrypoint
ENTRYPOINT ["python", "/app/docker_agent.py"]
