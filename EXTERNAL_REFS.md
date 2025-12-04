# External Library Reference Guide

**Project**: SWE-Bench Plugin Efficiency Benchmark Tool
**Purpose**: Quick reference for all external libraries used in the benchmark harness
**Last Updated**: December 2024

---

## Table of Contents

1. [Claude Agent SDK](#1-claude-agent-sdk)
2. [Pydantic](#2-pydantic)
3. [HuggingFace Datasets](#3-huggingface-datasets)
4. [Textual](#4-textual-tui-framework)
5. [Quick Reference Tables](#5-quick-reference-tables)

---

## 1. Claude Agent SDK

**Package**: `claude-agent-sdk`
**Install**: `pip install claude-agent-sdk`
**Purpose**: Execute Claude agents with tool use, token tracking, and MCP server integration

### 1.1 Core Imports

```python
from claude_agent_sdk import (
    query,                    # Main query function
    ClaudeSDKClient,          # Multi-turn client
    ClaudeAgentOptions,       # Configuration options
    AssistantMessage,         # Response message type
    ResultMessage,            # Final result type
    TextBlock,                # Text content block
    tool,                     # Tool decorator
    create_sdk_mcp_server,    # Create MCP server
    CLINotFoundError,         # Error types
    ProcessError,
    CLIJSONDecodeError
)
```

### 1.2 Basic Agent Query

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def run_agent():
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash", "Edit", "Glob", "Grep"],
        permission_mode="acceptEdits",  # "default" | "acceptEdits" | "bypassPermissions"
        cwd="/path/to/project",
        max_turns=10,
        max_budget_usd=5.0,
        model="claude-sonnet-4-5"
    )

    async for message in query(prompt="Analyze this project", options=options):
        print(message)

asyncio.run(run_agent())
```

### 1.3 Multi-Turn Conversation

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

async def multi_turn():
    options = ClaudeAgentOptions(allowed_tools=["Read", "Write", "Bash"])

    async with ClaudeSDKClient(options=options) as client:
        # First turn
        await client.query("Create a hello.py file")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text)

        # Second turn (maintains context)
        await client.query("What's in that file?")
        async for msg in client.receive_response():
            # Process response...
            pass
```

### 1.4 Token Usage Tracking

```python
from claude_agent_sdk import query, AssistantMessage, ResultMessage

async def track_tokens(prompt: str):
    total_input = 0
    total_output = 0
    total_cost = 0.0

    async for message in query(prompt=prompt):
        if isinstance(message, AssistantMessage) and hasattr(message, 'usage'):
            total_input += message.usage.get('input_tokens', 0)
            total_output += message.usage.get('output_tokens', 0)

        if isinstance(message, ResultMessage):
            total_cost = getattr(message, 'total_cost_usd', 0.0)

    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_cost_usd": total_cost
    }
```

### 1.5 Custom Tool Definition

```python
from claude_agent_sdk import tool, create_sdk_mcp_server, query, ClaudeAgentOptions
from typing import Any

@tool("run_test", "Execute a test", {"test_name": str, "timeout": int})
async def run_test(args: dict[str, Any]) -> dict[str, Any]:
    test_name = args.get("test_name")
    timeout = args.get("timeout", 60)

    # Execute test logic here
    result = f"Test '{test_name}' passed"

    return {
        "content": [{"type": "text", "text": result}]
    }

# Create MCP server with custom tools
custom_server = create_sdk_mcp_server(
    name="benchmark-tools",
    version="1.0.0",
    tools=[run_test]
)

# Use with agent
async def use_custom_tools():
    options = ClaudeAgentOptions(
        mcp_servers={"benchmark": custom_server},
        allowed_tools=["mcp__benchmark__run_test"]
    )
    async for message in query(prompt="Run the auth test", options=options):
        print(message)
```

### 1.6 MCP Server Configuration

```python
# External MCP servers (stdio)
options = ClaudeAgentOptions(
    mcp_servers={
        "filesystem": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "mcp_server_filesystem"],
            "env": {"ALLOWED_PATHS": "/home/projects"}
        },
        "bash": {
            "type": "stdio",
            "command": "npx",
            "args": ["@modelcontextprotocol/server-bash"]
        }
    },
    allowed_tools=["mcp__filesystem__read", "mcp__bash__execute"]
)
```

### 1.7 Error Handling

```python
from claude_agent_sdk import query, CLINotFoundError, ProcessError, CLIJSONDecodeError

async def safe_query(prompt: str):
    try:
        async for message in query(prompt=prompt):
            yield message
    except CLINotFoundError:
        raise RuntimeError("Claude Code CLI not installed")
    except ProcessError as e:
        raise RuntimeError(f"Process failed: exit code {e.exit_code}")
    except CLIJSONDecodeError as e:
        raise RuntimeError(f"Parse error: {e.original_error}")
```

### 1.8 Timeout Protection

```python
import asyncio
from claude_agent_sdk import query

async def query_with_timeout(prompt: str, timeout_sec: int = 300):
    try:
        async with asyncio.timeout(timeout_sec):
            async for message in query(prompt=prompt):
                yield message
    except asyncio.TimeoutError:
        raise RuntimeError(f"Query timed out after {timeout_sec}s")
```

---

## 2. Pydantic

**Package**: `pydantic`
**Install**: `pip install pydantic`
**Purpose**: Configuration validation and type-safe data models

### 2.1 Core Imports

```python
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Optional, List, Dict, Literal
```

### 2.2 Basic Model Definition

```python
from pydantic import BaseModel, Field, ConfigDict

class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_default=True)

    source: str = Field(description="Dataset source path")
    split: str = Field(default="test[:10]", description="Dataset split")
    seed: int = Field(default=42, ge=0, description="Random seed")
    cache_dir: str = Field(default="~/.cache/swe-bench")
```

### 2.3 Field Constraints

```python
from pydantic import BaseModel, Field

class ExecutionConfig(BaseModel):
    runs_per_instance: int = Field(ge=1, le=100, default=5)
    max_parallel_tasks: int = Field(ge=1, default=4)
    timeout_per_run_sec: int = Field(ge=1, le=3600, default=900)
    temperature: float = Field(ge=0.0, le=1.0, default=0.2)
```

### 2.4 Nested Models

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PluginConfig(BaseModel):
    id: str
    name: str
    mcp_servers: Dict[str, str] = Field(default_factory=dict)
    allowed_tools: List[str] = Field(default_factory=list)

class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    execution: ExecutionConfig
    configs: List[PluginConfig]
    output_dir: str = Field(default="./results")
```

### 2.5 Custom Validators

```python
from pydantic import BaseModel, field_validator

class ModelConfig(BaseModel):
    name: str
    temperature: float = 0.2

    @field_validator('name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        valid_models = ["claude-sonnet-4-5", "claude-3-5-sonnet-20241022"]
        if v not in valid_models:
            raise ValueError(f"Invalid model: {v}")
        return v
```

### 2.6 YAML Loading

```python
import yaml
from pydantic import BaseModel

class Config(BaseModel):
    # ... fields ...

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.dump(self.model_dump(exclude_none=True), default_flow_style=False)
```

### 2.7 Serialization

```python
config = ExperimentConfig(...)

# To dict
config_dict = config.model_dump()
config_dict = config.model_dump(exclude_none=True)
config_dict = config.model_dump(exclude_unset=True)

# To JSON
json_str = config.model_dump_json(indent=2)
```

---

## 3. HuggingFace Datasets

**Package**: `datasets`
**Install**: `pip install datasets`
**Purpose**: Load and process SWE-bench dataset from HuggingFace Hub

### 3.1 Core Imports

```python
from datasets import load_dataset, Dataset, DatasetDict
```

### 3.2 Load Dataset

```python
from datasets import load_dataset

# Load with specific split
dataset = load_dataset(
    "princeton-nlp/SWE-bench-lite",
    split="test[:20]",
    cache_dir="~/.cache/swe-bench"
)
```

### 3.3 Split Slicing Syntax

```python
# Absolute indexing
dataset = load_dataset("...", split="test[:20]")      # First 20
dataset = load_dataset("...", split="test[10:20]")    # Rows 10-19
dataset = load_dataset("...", split="test[-50:]")     # Last 50

# Percentage slicing
dataset = load_dataset("...", split="train[:10%]")    # First 10%
dataset = load_dataset("...", split="train[50%:60%]") # 50% to 60%

# Combining splits
dataset = load_dataset("...", split="train[:10%]+train[-10%:]")
```

### 3.4 Access Fields

```python
# Column access
all_ids = dataset["instance_id"]
first_id = dataset["instance_id"][0]

# Row access
first_row = dataset[0]
first_three = dataset[:3]

# Column metadata
columns = dataset.column_names  # ['instance_id', 'repo', ...]
num_rows = len(dataset)
```

### 3.5 Iteration Patterns

```python
# Simple iteration
for example in dataset:
    print(example['instance_id'])
    print(example['problem_statement'])

# With index
for i, example in enumerate(dataset):
    print(f"{i}: {example['instance_id']}")

# Batched processing
def process_batch(examples):
    return {'processed': [f"done_{x}" for x in examples['instance_id']]}

dataset = dataset.map(process_batch, batched=True, batch_size=32)
```

### 3.6 Filtering

```python
# Filter examples
filtered = dataset.filter(lambda x: 'django' in x['repo'])

# Select columns
subset = dataset.select_columns(['instance_id', 'problem_statement'])
```

### 3.7 Cache Control

```python
from datasets import load_dataset, disable_caching

# Custom cache directory
dataset = load_dataset("...", cache_dir="/path/to/cache")

# Force redownload
dataset = load_dataset("...", download_mode='force_redownload')

# Disable caching globally
disable_caching()
```

---

## 4. Textual (TUI Framework)

**Package**: `textual`
**Install**: `pip install textual`
**Purpose**: Terminal User Interface for real-time benchmark monitoring

### 4.1 Core Imports

```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, ProgressBar, Static, Label
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.binding import Binding
from textual.worker import work, get_current_worker
from textual.reactive import reactive
```

### 4.2 Basic App Structure

```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable

class BenchmarkApp(App):
    CSS = """
    Screen { layout: vertical; }
    DataTable { height: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "save", "Save"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="results")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Config", "Success", "Time", "Tokens")

    def action_save(self) -> None:
        self.notify("Results saved")

if __name__ == "__main__":
    BenchmarkApp().run()
```

### 4.3 DataTable Widget

```python
from textual.widgets import DataTable
from rich.text import Text

# Setup
table = self.query_one(DataTable)
table.cursor_type = "row"
table.zebra_stripes = True
table.add_columns("Config", "Status", "Progress", "Time")

# Add rows
table.add_row("config_a", Text("PASS", style="green"), "100%", "42.1s", key="config_a")

# Update cell
row_idx = list(table.row_keys).index("config_a")
table.update_cell("Status", row_idx, Text("RUNNING", style="yellow"))

# Remove row
table.remove_row("config_a")
```

### 4.4 ProgressBar Widget

```python
from textual.widgets import ProgressBar

# Create
progress = ProgressBar(total=100, id="main_progress")

# Update
progress.update(completed=50)  # 50%
progress.update(total=200, completed=100)  # Change total
```

### 4.5 Background Workers with Thread Safety

```python
from textual.worker import work, get_current_worker

class BenchmarkApp(App):
    @work(thread=True)
    def run_benchmark(self, config_name: str):
        """Run in background thread."""
        worker = get_current_worker()

        for step in range(100):
            if worker.is_cancelled:
                return

            # Do work...
            time.sleep(0.1)

            # Update UI safely from thread
            self.call_from_thread(self.update_progress, config_name, step + 1)

        self.call_from_thread(self.on_complete, config_name)

    def update_progress(self, config: str, step: int):
        """Called on main thread."""
        progress = self.query_one(f"#progress-{config}", ProgressBar)
        progress.update(completed=step)

    def action_start(self):
        self.run_benchmark("config_a")  # Launches worker
```

### 4.6 Keyboard Bindings

```python
from textual.binding import Binding

class MyApp(App):
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("s", "start", "Start", show=True),
        Binding("space", "pause", "Pause", show=True),
        Binding("r", "reset", "Reset", show=True),
    ]

    def action_start(self) -> None:
        self.notify("Started")

    def action_pause(self) -> None:
        self.is_paused = not self.is_paused
```

### 4.7 CSS Styling

```python
class MyApp(App):
    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #left-panel {
        width: 50%;
        border: solid $primary;
        padding: 1;
    }

    #right-panel {
        width: 50%;
        border: solid $primary;
    }

    ProgressBar {
        margin: 1 0;
        height: 1;
    }

    DataTable {
        height: 1fr;
    }

    .status-running { color: $warning; }
    .status-pass { color: $success; }
    .status-fail { color: $error; }
    """
```

### 4.8 Layout Containers

```python
from textual.containers import Container, Vertical, Horizontal

def compose(self) -> ComposeResult:
    yield Header()

    with Container(id="main"):
        with Horizontal():
            with Vertical(id="left-panel"):
                yield Label("Progress")
                yield ProgressBar(total=100)

            with Vertical(id="right-panel"):
                yield Label("Results")
                yield DataTable()

    yield Footer()
```

---

## 5. Quick Reference Tables

### 5.1 Claude Agent SDK Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `allowed_tools` | `List[str]` | `[]` | Tool whitelist |
| `permission_mode` | `str` | `"default"` | `"default"`, `"acceptEdits"`, `"bypassPermissions"` |
| `max_turns` | `int` | `None` | Max conversation turns |
| `max_budget_usd` | `float` | `None` | Cost limit |
| `model` | `str` | `None` | Model ID |
| `cwd` | `Path` | `None` | Working directory |
| `mcp_servers` | `Dict` | `{}` | MCP server configs |
| `system_prompt` | `str` | `None` | System instructions |

### 5.2 Pydantic Field Constraints

| Constraint | Type | Example |
|------------|------|---------|
| `ge` | number | `Field(ge=0)` - greater or equal |
| `le` | number | `Field(le=100)` - less or equal |
| `gt` | number | `Field(gt=0)` - greater than |
| `lt` | number | `Field(lt=1.0)` - less than |
| `min_length` | string | `Field(min_length=1)` |
| `max_length` | string | `Field(max_length=255)` |
| `pattern` | string | `Field(pattern=r'^[a-z]+$')` |

### 5.3 HuggingFace Split Syntax

| Syntax | Example | Description |
|--------|---------|-------------|
| `[:N]` | `test[:20]` | First N rows |
| `[N:]` | `test[10:]` | From row N to end |
| `[N:M]` | `test[10:20]` | Rows N to M-1 |
| `[-N:]` | `test[-50:]` | Last N rows |
| `[:N%]` | `train[:10%]` | First N percent |
| `[N%:M%]` | `train[50%:60%]` | Percentage range |

### 5.4 Textual Widget Methods

| Widget | Method | Description |
|--------|--------|-------------|
| `DataTable` | `add_columns(*names)` | Add column headers |
| `DataTable` | `add_row(*values, key=...)` | Add data row |
| `DataTable` | `update_cell(col, row, value)` | Update cell |
| `DataTable` | `remove_row(key)` | Remove row by key |
| `ProgressBar` | `update(completed=N)` | Set progress |
| `App` | `call_from_thread(fn, *args)` | Thread-safe UI call |
| `App` | `notify(message)` | Show notification |
| `App` | `query_one(selector)` | Find widget |

### 5.5 Token Cost Calculation

```python
# Claude Sonnet pricing (approximate)
INPUT_COST_PER_TOKEN = 0.000003    # $3 per 1M tokens
OUTPUT_COST_PER_TOKEN = 0.000015   # $15 per 1M tokens
CACHE_READ_COST_PER_TOKEN = 0.0000003  # $0.30 per 1M tokens

def calculate_cost(usage: dict) -> float:
    input_cost = usage.get('input_tokens', 0) * INPUT_COST_PER_TOKEN
    output_cost = usage.get('output_tokens', 0) * OUTPUT_COST_PER_TOKEN
    cache_cost = usage.get('cache_read_input_tokens', 0) * CACHE_READ_COST_PER_TOKEN
    return input_cost + output_cost + cache_cost
```

---

## Installation Summary

```bash
# Core dependencies
pip install claude-agent-sdk
pip install pydantic
pip install datasets
pip install textual

# Additional
pip install pyyaml      # YAML config parsing
pip install aiohttp     # Async HTTP for custom tools
```

---

## File Structure Reference

```
swe-bench-plugin-harness/
├── src/
│   ├── config.py           # Pydantic models (Section 2)
│   ├── dataset_loader.py   # HuggingFace integration (Section 3)
│   ├── runner.py           # Claude Agent SDK (Section 1)
│   ├── metrics.py          # Token tracking (Section 1.4)
│   └── tui.py              # Textual UI (Section 4)
├── EXTERNAL_REFS.md        # This file
└── requirements.txt
```
