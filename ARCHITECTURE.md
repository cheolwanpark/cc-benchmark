# Architecture

**Project**: SWE-Bench Plugin Efficiency Benchmark Tool
**Version**: 1.0.0

---

## Overview

A benchmarking harness that measures plugin/tool efficiency on SWE-bench across four dimensions:

- **Success Rate** - solve rate percentage
- **Execution Time** - seconds per task
- **Tool Call Efficiency** - API calls per task
- **Token Usage** - input + output tokens, cost projection

---

## Project Structure

```
cc-benchmark/
├── pyproject.toml
├── swe_bench_harness/
│   ├── __init__.py
│   ├── cli.py                # CLI entry + Rich progress
│   ├── config.py             # Pydantic models + YAML loading
│   ├── dataset.py            # HuggingFace loader + Instance model
│   ├── agent.py              # Claude SDK wrapper (single task)
│   ├── runner.py             # Benchmark orchestration loop
│   ├── metrics.py            # RunRecord + aggregation + stats
│   ├── reporter.py           # JSON/YAML/HTML export
│   └── templates/
│       └── report.html       # Chart.js template
├── tests/
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_runner.py
│   └── test_metrics.py
└── examples/
    └── experiment.yaml
```

---

## Module Specifications

### cli.py

**Purpose**: Command-line interface with Rich-based progress display.

**Responsibilities**:
- Parse command-line arguments (config path, dry-run, output dir)
- Load and validate configuration
- Launch benchmark runner with progress display
- Print summary table on completion

**Key Dependencies**: `argparse`, `rich`

**Public Interface**:
```python
def main() -> int:
    """CLI entry point. Returns exit code."""

def print_summary_table(results: BenchmarkResults) -> None:
    """Display final results as Rich table."""
```

---

### config.py

**Purpose**: Configuration management with Pydantic validation.

**Responsibilities**:
- Define typed configuration models
- Load and validate YAML files
- Provide sensible defaults

**Key Dependencies**: `pydantic`, `pyyaml`

**Models**:
```python
class DatasetConfig(BaseModel):
    source: str = "princeton-nlp/SWE-bench-lite"
    split: str = "test[:10]"
    seed: int = 42
    cache_dir: str = "~/.cache/swe-bench"

class ExecutionConfig(BaseModel):
    runs_per_instance: int = Field(default=5, ge=1, le=100)
    max_parallel_tasks: int = Field(default=4, ge=1)
    timeout_per_run_sec: int = Field(default=900, ge=1)

class ModelConfig(BaseModel):
    name: str = "claude-sonnet-4-5"
    max_tokens: int = 4096
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)

class PluginConfig(BaseModel):
    id: str
    name: str
    description: str = ""
    mcp_servers: Dict[str, Any] = Field(default_factory=dict)
    allowed_tools: List[str] = Field(default_factory=list)

class ExperimentConfig(BaseModel):
    name: str
    dataset: DatasetConfig
    execution: ExecutionConfig
    model: ModelConfig
    configs: List[PluginConfig]
    output_dir: str = "./results"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig": ...
```

---

### dataset.py

**Purpose**: Load SWE-bench instances from HuggingFace.

**Responsibilities**:
- Fetch dataset from HuggingFace Hub
- Parse into typed dataclass instances
- Handle caching for repeated runs

**Key Dependencies**: `datasets` (HuggingFace)

**Public Interface**:
```python
@dataclass
class SWEBenchInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    test_cmd: str

class DatasetLoader:
    def load(self, config: DatasetConfig) -> List[SWEBenchInstance]: ...
```

---

### agent.py

**Purpose**: Execute single task with Claude Agent SDK.

**Responsibilities**:
- Create Claude agent session with configured tools
- Execute agent until completion or timeout
- Track token usage and tool calls
- Return structured execution result

**Key Dependencies**: `claude-agent-sdk`

**Public Interface**:
```python
@dataclass
class ExecutionResult:
    success: bool
    duration_sec: float
    tokens_input: int
    tokens_output: int
    tool_calls_total: int
    tool_calls_by_name: Dict[str, int]
    error_reason: Optional[str] = None
    cost_usd: Optional[float] = None

class ClaudeAgent:
    def __init__(self, model_config: ModelConfig, plugin_config: PluginConfig): ...

    async def execute(
        self,
        instance: SWEBenchInstance,
        timeout_sec: int
    ) -> ExecutionResult: ...
```

---

### runner.py

**Purpose**: Orchestrate benchmark execution across configs and instances.

**Responsibilities**:
- Iterate over plugin configs × instances × runs
- Manage concurrent execution with semaphore
- Collect results and emit progress events
- Handle Docker environment setup/cleanup

**Key Dependencies**: `asyncio`, `agent.py`, `dataset.py`, `metrics.py`

**Public Interface**:
```python
class BenchmarkRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        instances: List[SWEBenchInstance]
    ): ...

    @property
    def total_runs(self) -> int: ...

    async def run(self) -> AsyncIterator[RunRecord]:
        """Yield results as they complete."""

    def get_results(self) -> BenchmarkResults: ...
```

---

### metrics.py

**Purpose**: Collect run records and compute aggregate statistics.

**Responsibilities**:
- Define RunRecord dataclass
- Aggregate results by config
- Compute mean, p50, p90, std for metrics
- Calculate efficiency scores

**Key Dependencies**: `statistics`, `dataclasses`

**Public Interface**:
```python
@dataclass
class RunRecord:
    run_id: str
    instance_id: str
    config_id: str
    timestamp: datetime
    success: bool
    duration_sec: float
    tokens_input: int
    tokens_output: int
    tool_calls_total: int
    tool_calls_by_name: Dict[str, int]
    error_reason: Optional[str] = None
    cost_usd: Optional[float] = None

@dataclass
class ConfigSummary:
    config_id: str
    config_name: str
    total_runs: int
    success_count: int
    success_rate: float
    duration_mean: float
    duration_p50: float
    duration_p90: float
    tokens_mean: int
    tool_calls_mean: float
    cost_mean: float

@dataclass
class BenchmarkResults:
    experiment_name: str
    timestamp: datetime
    total_duration_sec: float
    records: List[RunRecord]
    summaries: List[ConfigSummary]

class MetricsAggregator:
    def aggregate(self, records: List[RunRecord]) -> List[ConfigSummary]: ...
```

---

### reporter.py

**Purpose**: Export results to JSON, YAML, and HTML formats.

**Responsibilities**:
- Serialize results to JSON/YAML
- Render HTML report with Chart.js graphs
- Save all outputs to configured directory

**Key Dependencies**: `jinja2`, `json`, `pyyaml`

**Public Interface**:
```python
class Reporter:
    def __init__(self, output_dir: str): ...

    def generate_all(self, results: BenchmarkResults) -> Dict[str, Path]:
        """Generate all report formats. Returns {format: path}."""

    def generate_json(self, results: BenchmarkResults) -> Path: ...
    def generate_yaml(self, results: BenchmarkResults) -> Path: ...
    def generate_html(self, results: BenchmarkResults) -> Path: ...
```

---

### templates/report.html

**Purpose**: Self-contained HTML template with Chart.js visualization.

**Features**:
- Summary statistics table
- Success rate bar chart
- Cost comparison bar chart
- Token usage chart
- No external CSS dependencies (inline styles)

**Template Variables**:
```
experiment_name: str
timestamp: str
configs: List[ConfigSummary]
chart_data: {
    labels: List[str],
    success_rates: List[float],
    costs: List[float],
    tokens: List[int]
}
```

---

## Data Flow

```
┌─────────┐     ┌──────────┐     ┌───────────┐
│ cli.py  │────▶│ config.py│────▶│ dataset.py│
└────┬────┘     └──────────┘     └─────┬─────┘
     │                                 │
     │         ┌──────────────────────┘
     │         │
     ▼         ▼
┌─────────────────┐     ┌──────────┐
│    runner.py    │────▶│ agent.py │
└────────┬────────┘     └──────────┘
         │
         ▼
┌─────────────────┐
│   metrics.py    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  reporter.py    │──────▶ JSON / YAML / HTML
└─────────────────┘
```

---

## Execution Flow

```
1. CLI STARTUP
   ├─ Parse arguments (--config, --output-dir, --dry-run)
   ├─ Load YAML → validate with Pydantic
   └─ Initialize Rich progress display

2. DATASET LOADING
   ├─ Check local cache
   ├─ Fetch from HuggingFace if needed
   └─ Parse into SWEBenchInstance list

3. BENCHMARK LOOP
   for each plugin_config in configs:
       for each instance in dataset:
           for run_num in range(runs_per_instance):
               ├─ Create ClaudeAgent with plugin_config
               ├─ Execute agent on instance
               ├─ Collect ExecutionResult
               ├─ Create RunRecord
               └─ Update progress display

4. AGGREGATION
   ├─ Group records by config_id
   ├─ Compute statistics (mean, p50, p90, std)
   └─ Calculate efficiency scores

5. REPORTING
   ├─ Generate report.json
   ├─ Generate report.yaml
   ├─ Generate report.html (with charts)
   └─ Print summary table to console
```

---

## Dependencies

### Core

| Package | Version | Purpose |
|---------|---------|---------|
| `claude-agent-sdk` | latest | Claude agent execution |
| `pydantic` | ^2.0 | Configuration validation |
| `datasets` | ^2.0 | HuggingFace dataset loading |
| `rich` | ^13.0 | CLI progress display |
| `jinja2` | ^3.0 | HTML template rendering |
| `pyyaml` | ^6.0 | YAML config parsing |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Testing |
| `pytest-asyncio` | Async test support |
| `ruff` | Linting |

---

## Design Decisions

### 1. Rich over Textual for Progress Display

**Decision**: Use Rich progress bars instead of full Textual TUI.

**Rationale**:
- Benchmark runs are non-interactive (start → wait → results)
- Rich provides sufficient feedback (progress bar, live status)
- Simpler implementation (~80 lines vs ~300 lines)
- Lower learning curve for contributors

### 2. Chart.js via CDN for HTML Reports

**Decision**: Use Chart.js loaded from CDN instead of generating static images.

**Rationale**:
- No matplotlib/pillow dependencies
- Interactive charts (hover, zoom)
- Single HTML file output (self-contained)
- Template remains simple (~50 lines)

### 3. Separate agent.py from runner.py

**Decision**: Keep agent execution separate from orchestration.

**Rationale**:
- Clear single responsibility
- Agent can be tested in isolation
- Easier to swap agent implementations
- Runner focuses on concurrency/scheduling

### 4. Flat Module Structure

**Decision**: Single-level package without nested directories.

**Rationale**:
- 7 modules is manageable without subpackages
- Simpler imports (`from swe_bench_harness import config`)
- Easier navigation for new contributors
- No circular import risks from deep nesting

---

## Testing Strategy

| Module | Test Focus |
|--------|------------|
| `config.py` | YAML parsing, validation errors, defaults |
| `dataset.py` | HF loading, caching, instance parsing |
| `agent.py` | Mock Claude SDK, token tracking, timeout |
| `runner.py` | Concurrency, progress events, error handling |
| `metrics.py` | Aggregation math, edge cases (empty, single) |
| `reporter.py` | Output format correctness, file creation |

---

## Future Considerations

Items explicitly out of scope for v1.0:

- ❌ Multiple LLM support (fixed to Claude)
- ❌ Distributed execution across machines
- ❌ Real-time cost billing integration
- ❌ Synthetic dataset generation

Potential v1.1 additions:

- ⏳ Result comparison between experiments
- ⏳ CSV export format
- ⏳ Webhook notifications on completion
