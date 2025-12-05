"""NDJSON protocol for Docker container-host communication.

This module defines the message types exchanged between the Docker container
running the Claude agent and the host benchmark runner. All messages use
newline-delimited JSON (NDJSON) format with explicit sequence numbers.

Container writes to stdout (protocol channel only); SDK logs go to stderr.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of protocol messages."""

    INIT = "init"
    ASSISTANT = "assistant"
    RESULT = "result"
    ERROR = "error"


class ErrorType(str, Enum):
    """Types of errors that can occur in the container."""

    CONFIG_MISSING = "config_missing"
    CONFIG_INVALID = "config_invalid"
    CLI_NOT_FOUND = "cli_not_found"
    SDK_ERROR = "sdk_error"
    CLI_CRASH = "cli_crash"  # SIGSEGV and similar CLI crashes
    GIT_ERROR = "git_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


# =============================================================================
# Protocol Message Payloads
# =============================================================================


class InitPayload(BaseModel):
    """Payload for init message (container startup)."""

    type: Literal["init"] = "init"
    seq: int
    timestamp: str
    session_id: str | None = None
    model: str | None = None


class ToolCallInfo(BaseModel):
    """Information about a tool call."""

    name: str
    id: str


class AssistantPayload(BaseModel):
    """Payload for assistant message (during execution)."""

    type: Literal["assistant"] = "assistant"
    seq: int
    timestamp: str
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    text_preview: str | None = None  # First 200 chars of text content


class UsageInfo(BaseModel):
    """Token usage information."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class ResultPayload(BaseModel):
    """Payload for final result message."""

    type: Literal["result"] = "result"
    seq: int
    timestamp: str
    success: bool
    usage: UsageInfo = Field(default_factory=UsageInfo)
    cost_usd: float = 0.0
    tool_calls_total: int = 0
    tool_calls_by_name: dict[str, int] = Field(default_factory=dict)
    duration_sec: float = 0.0
    error_reason: str | None = None
    # Note: patch is in /output/patch.diff file, not in this message


class ErrorPayload(BaseModel):
    """Payload for error message."""

    type: Literal["error"] = "error"
    seq: int
    timestamp: str
    error_type: ErrorType
    error_message: str
    recoverable: bool = False
    traceback: str | None = None


# Union of all message payloads
ProtocolMessage = InitPayload | AssistantPayload | ResultPayload | ErrorPayload


# =============================================================================
# Serialization (Container Side)
# =============================================================================


class MessageSerializer:
    """Serialize messages to NDJSON format for stdout."""

    def __init__(self) -> None:
        self._sequence = 0

    def _next_seq(self) -> int:
        """Get next sequence number."""
        seq = self._sequence
        self._sequence += 1
        return seq

    def _timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def init(self, session_id: str | None = None, model: str | None = None) -> str:
        """Create init message."""
        msg = InitPayload(
            seq=self._next_seq(),
            timestamp=self._timestamp(),
            session_id=session_id,
            model=model,
        )
        return msg.model_dump_json()

    def assistant(
        self,
        tool_calls: list[tuple[str, str]] | None = None,
        text_preview: str | None = None,
    ) -> str:
        """Create assistant message.

        Args:
            tool_calls: List of (name, id) tuples for tool calls
            text_preview: Preview of text content (truncated to 200 chars)
        """
        calls = [ToolCallInfo(name=name, id=id_) for name, id_ in (tool_calls or [])]
        msg = AssistantPayload(
            seq=self._next_seq(),
            timestamp=self._timestamp(),
            tool_calls=calls,
            text_preview=text_preview[:200] if text_preview else None,
        )
        return msg.model_dump_json()

    def result(
        self,
        success: bool,
        usage: UsageInfo | None = None,
        cost_usd: float = 0.0,
        tool_calls_total: int = 0,
        tool_calls_by_name: dict[str, int] | None = None,
        duration_sec: float = 0.0,
        error_reason: str | None = None,
    ) -> str:
        """Create result message."""
        msg = ResultPayload(
            seq=self._next_seq(),
            timestamp=self._timestamp(),
            success=success,
            usage=usage or UsageInfo(),
            cost_usd=cost_usd,
            tool_calls_total=tool_calls_total,
            tool_calls_by_name=tool_calls_by_name or {},
            duration_sec=round(duration_sec, 2),
            error_reason=error_reason,
        )
        return msg.model_dump_json()

    def error(
        self,
        error_type: ErrorType,
        error_message: str,
        recoverable: bool = False,
        traceback: str | None = None,
    ) -> str:
        """Create error message."""
        msg = ErrorPayload(
            seq=self._next_seq(),
            timestamp=self._timestamp(),
            error_type=error_type,
            error_message=error_message,
            recoverable=recoverable,
            traceback=traceback,
        )
        return msg.model_dump_json()


# =============================================================================
# Deserialization (Host Side)
# =============================================================================


class ParseError(Exception):
    """Error parsing protocol message."""

    def __init__(self, line: str, reason: str) -> None:
        self.line = line
        self.reason = reason
        super().__init__(f"Failed to parse message: {reason}")


def parse_message(line: str) -> ProtocolMessage:
    """Parse a single NDJSON line into a typed message.

    Args:
        line: JSON string (may include trailing newline)

    Returns:
        Typed protocol message

    Raises:
        ParseError: If the line cannot be parsed
    """
    line = line.strip()
    if not line:
        raise ParseError(line, "Empty line")

    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        raise ParseError(line, f"Invalid JSON: {e}") from e

    msg_type = data.get("type")
    if not msg_type:
        raise ParseError(line, "Missing 'type' field")

    try:
        if msg_type == "init":
            return InitPayload.model_validate(data)
        elif msg_type == "assistant":
            return AssistantPayload.model_validate(data)
        elif msg_type == "result":
            return ResultPayload.model_validate(data)
        elif msg_type == "error":
            return ErrorPayload.model_validate(data)
        else:
            raise ParseError(line, f"Unknown message type: {msg_type}")
    except Exception as e:
        raise ParseError(line, f"Validation error: {e}") from e


# =============================================================================
# Aggregated Metrics (Host Side)
# =============================================================================


class AggregatedMetrics:
    """Accumulate metrics from protocol messages.

    Used by the host to build ExecutionResult from container output.
    """

    def __init__(self) -> None:
        self.messages_received: int = 0
        self.tool_calls_total: int = 0
        self.tool_calls_by_name: dict[str, int] = {}
        self.usage: UsageInfo = UsageInfo()
        self.cost_usd: float = 0.0
        self.duration_sec: float = 0.0
        self.success: bool = False
        self.error_reason: str | None = None
        self.errors: list[ErrorPayload] = []

    def process(self, msg: ProtocolMessage) -> None:
        """Process a message and update metrics."""
        self.messages_received += 1

        if isinstance(msg, AssistantPayload):
            # Track tool calls
            for call in msg.tool_calls:
                self.tool_calls_total += 1
                self.tool_calls_by_name[call.name] = (
                    self.tool_calls_by_name.get(call.name, 0) + 1
                )

        elif isinstance(msg, ResultPayload):
            # Extract final metrics
            self.success = msg.success
            self.usage = msg.usage
            self.cost_usd = msg.cost_usd
            self.tool_calls_total = msg.tool_calls_total
            self.tool_calls_by_name = msg.tool_calls_by_name
            self.duration_sec = msg.duration_sec
            self.error_reason = msg.error_reason

        elif isinstance(msg, ErrorPayload):
            self.errors.append(msg)
            if not msg.recoverable:
                self.success = False
                self.error_reason = msg.error_message


# =============================================================================
# Container Config Schema
# =============================================================================


class InstanceConfig(BaseModel):
    """SWE-bench instance configuration."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    FAIL_TO_PASS: str = ""
    PASS_TO_PASS: str = ""


class ModelConfigSchema(BaseModel):
    """Model configuration for the agent."""

    name: str


class BenchmarkConfigSchema(BaseModel):
    """Benchmark configuration for the agent."""

    allowed_tools: list[str] = Field(default_factory=list)
    plugins: list[dict[str, Any]] = Field(default_factory=list)
    envs: dict[str, str] = Field(default_factory=dict)


class ExecutionConfigSchema(BaseModel):
    """Execution settings."""

    timeout_sec: int = 600
    max_turns: int = 50


class ContainerConfig(BaseModel):
    """Full configuration passed to container via /config/config.json."""

    version: str = "1.0"
    instance: InstanceConfig
    model: ModelConfigSchema
    benchmark: BenchmarkConfigSchema = Field(default_factory=BenchmarkConfigSchema)
    execution: ExecutionConfigSchema = Field(default_factory=ExecutionConfigSchema)
