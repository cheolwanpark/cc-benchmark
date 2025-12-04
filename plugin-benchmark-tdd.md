# SWE-Bench Plugin Efficiency Benchmark Tool
## Technical Design Document

**Project Name**: swe-bench-plugin-harness  
**Version**: 1.0.0-design  
**Date**: December 2024  
**Audience**: Open-source contributors, Claude plugin developers  
**Purpose**: Automated benchmarking framework for measuring plugin/tool efficiency on SWE-bench

---

## 1. Executive Summary

This document specifies a **production-ready benchmarking harness** that measures code-generation plugin efficiency across four dimensions:

- ✅ **Success Rate** (solve rate, %)
- ⏱️ **Execution Time** (seconds per task)
- 🔧 **Tool Call Efficiency** (API calls per task)
- 💰 **Token Usage** (input + output tokens, cost projection)

The tool automates repeated execution of SWE-bench instances across different **plugin configurations** (ablation studies), collects detailed metrics, and presents results via a TUI dashboard and exportable JSON/YAML reports.

**Key Innovation**: Unlike existing benchmarks that compare models (GPT-4 vs Claude), this harness compares **plugin combinations** (BM25 search vs dense search, full test suite vs minimal test) under identical model conditions.

---

## 2. Project Goals & Scope

### 2.1 Goals

1. **Enable plugin optimization**: Quantify the efficiency trade-offs when adding/removing tools
2. **Facilitate reproducible research**: Same YAML config + dataset = identical results across machines
3. **Reduce development cost**: Avoid expensive trial-and-error by pre-testing plugin combinations
4. **Support open-source ecosystem**: Make benchmarking accessible to independent plugin developers

### 2.2 Out-of-Scope

- ❌ Evaluating multiple LLMs simultaneously (fixed: Claude 3.5 Sonnet or user-specified model)
- ❌ Synthetic data generation or dataset creation
- ❌ Real-time cost billing integration (only local approximation)
- ❌ Distributed benchmarking across multiple machines (future version)

### 2.3 Success Criteria

- Runs 10 instances × 5 repeats = 50 benchmark iterations in <2 hours (single machine, 8-core)
- Reports success rate, execution time, tool calls, and tokens with <2% variance between runs
- YAML configuration is human-readable and requires <5 minutes to set up a new ablation study
- TUI provides real-time feedback on >95% of runs before completion

---

## 3. Architecture Overview

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────┐
│              Main Entry Point (CLI)                      │
│          benchmark.py --config experiments.yaml          │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐   ┌────▼─────┐  ┌────▼───────┐
   │ Dataset │   │  Runner  │  │   Config   │
   │ Loader  │   │  Engine  │  │  Parser    │
   │(HF)     │   │(Async)   │  │(YAML/Py)   │
   └────┬────┘   └────┬─────┘  └────┬───────┘
        │             │             │
        │   ┌─────────▼─────────┐   │
        │   │  Runner Executor  │◄──┘
        │   │ (Claude Agent SDK)│
        │   └─────────┬─────────┘
        │             │
        └─────────────┼─────────────────────┐
                      │                     │
           ┌──────────▼─────────┐  ┌────────▼──────┐
           │  Metrics Collector │  │  Environment  │
           │  (JSON/JSONL)      │  │  Manager      │
           │                    │  │  (Docker/Git) │
           └──────────┬─────────┘  └────────┬──────┘
                      │                     │
                      └─────────────┬───────┘
                                    │
                          ┌─────────▼────────┐
                          │  Report Generator│
                          │ (JSON/YAML/HTML) │
                          └─────────┬────────┘
                                    │
                          ┌─────────▼────────┐
                          │   TUI Display    │
                          │  (Textual)       │
                          └──────────────────┘
```

### 3.2 Execution Flow (High Level)

```
1. USER PROVIDES
   ├─ config.yaml (experiment definition)
   ├─ ANTHROPIC_API_KEY (environment variable)
   └─ --dry-run flag (optional)

2. SYSTEM STARTUP
   ├─ Parse YAML → validate config structure
   ├─ Load SWE-bench dataset from HuggingFace
   ├─ Initialize TUI + progress tracking
   └─ Prepare Docker environment (if required)

3. BENCHMARK EXECUTION (per config)
   for each plugin_config in configs:
       for each instance in dataset:
           for run_num in range(runs_per_instance):
               ├─ Create Claude Agent session
               ├─ Execute agent → collect tool calls, tokens
               ├─ Parse output → apply patch
               ├─ Run test suite → record pass/fail
               └─ Store metrics JSON

4. AGGREGATION & REPORTING
   ├─ Compute statistics (mean, p50, p90, std)
   ├─ Calculate efficiency scores (success/cost)
   ├─ Render TUI comparison table
   ├─ Export results to report.json / report.yaml
   └─ Generate HTML dashboard (optional)

5. USER INSPECTION
   ├─ View TUI results interactively
   └─ Download detailed JSON for post-analysis
```

---

## 4. Detailed Component Specifications

### 4.1 Dataset Loading Module (`dataset_loader.py`)

**Responsibility**: Load SWE-bench data from HuggingFace and prepare instances for benchmarking.

#### 4.1.1 Data Source

- **Dataset**: `princeton-nlp/SWE-bench-lite` (smaller, default) or `princeton-nlp/SWE-bench` (full)
- **Splits Available**:
  - `test`: Official test set (~300 instances)
  - `validation`: Validation set (~500 instances)
  - `mini`: Curated small set for quick testing (~10 instances)
- **Configuration via YAML**:
  ```yaml
  dataset:
    source: "princeton-nlp/SWE-bench-lite"
    split: "test[:20]"          # Load first 20 from test
    seed: 42                     # Reproducibility
    cache_dir: ~/.cache/swe-bench
  ```

#### 4.1.2 Instance Structure

Each instance loaded will have:

```python
@dataclass
class SWEBenchInstance:
    instance_id: str            # e.g., "django__django-16379"
    repo_url: str               # GitHub repo URL
    repo_name: str              # e.g., "django/django"
    base_commit: str            # Clean state commit
    test_patch: str             # Patch file (unified diff)
    problem_statement: str      # Issue description (user prompt)
    hints_text: str             # (Optional) hints for agent
    repo_type: str              # repo, api, or filesystem
    test_cmd: str               # Command to run tests
    # Computed during setup:
    test_files: List[str]       # Files modified by patch
    environment_setup_cmd: str  # Install dependencies
```

#### 4.1.3 Loading & Caching Strategy

```python
class DatasetLoader:
    def load(config: DatasetConfig) -> List[SWEBenchInstance]:
        """
        1. Check local cache (~/cache/swe-bench/<source>_<split>.pkl)
        2. If not found, fetch from HF via `datasets` library
        3. Parse fields into SWEBenchInstance dataclasses
        4. Save cache for future runs
        5. Return instances list
        
        Caching benefit: Subsequent runs avoid HF API calls
        """
        
    def validate_instance(instance: SWEBenchInstance) -> bool:
        """
        Checks:
        - repo_url is reachable
        - test_cmd is valid syntax
        - problem_statement is non-empty
        """
```

#### 4.1.4 Implementation Details

**Library**: `datasets` (HuggingFace)

```python
from datasets import load_dataset

dataset = load_dataset(
    "princeton-nlp/SWE-bench-lite",
    split="test",
    cache_dir="~/.cache/swe-bench"
)
# Returns HF Dataset object with lazy loading
```

**Key Consideration**: Ensure reproducibility via:
- Fixed `seed` in YAML (not random shuffling)
- Deterministic ordering (no `.shuffle()` by default)

---

### 4.2 Configuration Management (`config.py`)

**Responsibility**: Parse, validate, and provide type-safe access to YAML experiment configurations.

#### 4.2.1 Configuration Schema

```yaml
# experiments.yaml - Full Example
experiment:
  name: "swe-bench-plugin-ablation-v1"
  description: "Comparing BM25 vs Dense search efficiency"
  
  # Dataset configuration
  dataset:
    source: "princeton-nlp/SWE-bench-lite"
    split: "test[:20]"
    seed: 42

  # Execution parameters
  execution:
    runs_per_instance: 5        # N repeats per config/instance combo
    max_parallel_tasks: 4       # Concurrent instance runners
    timeout_per_run_sec: 900    # 15 minutes per run
    timeout_per_tool_call_sec: 60  # Individual tool timeout

  # Reproducibility
  random_seed: 42

# Claude API Configuration
model:
  name: "claude-3-5-sonnet-20241022"
  max_tokens: 4096
  temperature: 0.2              # Deterministic (not creative)
  top_p: 1.0

# Each plugin configuration to test
configs:
  - id: "baseline_no_plugins"
    name: "Baseline (No MCP Tools)"
    description: "Pure LLM without external tools"
    mcp_servers: {}
    allowed_tools: []
    expected_impact: "Low success rate, minimal tokens"

  - id: "with_file_editor"
    name: "File Editor Only"
    description: "Basic file read/write operations"
    mcp_servers:
      filesystem: "mcp://file-system"
    allowed_tools:
      - "mcp__filesystem__read_file"
      - "mcp__filesystem__write_file"
      - "mcp__filesystem__list_directory"
    expected_impact: "Moderate success, moderate tokens"

  - id: "full_stack"
    name: "Full Tool Stack"
    description: "All available tools enabled"
    mcp_servers:
      filesystem: "mcp://file-system"
      search: "mcp://repo-search-bm25"
      bash: "mcp://bash-execution"
      git: "mcp://git-operations"
    allowed_tools:
      - "mcp__*"  # Allow all MCP tools
    expected_impact: "High success, high tokens"

# Reporting settings
reporting:
  output_dir: "./results"
  report_format: ["json", "yaml", "html"]
  save_logs: true
  save_raw_traces: true       # Per-run Claude API traces
  log_level: "INFO"

# Optional: Cost estimation
pricing:
  model_input_cost_per_mtok: 3.0    # USD per million tokens
  model_output_cost_per_mtok: 15.0
```

#### 4.2.2 Configuration Classes (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class DatasetConfig(BaseModel):
    source: str = "princeton-nlp/SWE-bench-lite"
    split: str = "test[:10]"
    seed: int = 42
    cache_dir: str = "~/.cache/swe-bench"

class ExecutionConfig(BaseModel):
    runs_per_instance: int = Field(5, ge=1, le=100)
    max_parallel_tasks: int = Field(4, ge=1)
    timeout_per_run_sec: int = 900
    timeout_per_tool_call_sec: int = 60

class ModelConfig(BaseModel):
    name: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 1.0

class PluginConfig(BaseModel):
    id: str
    name: str
    description: str
    mcp_servers: Dict[str, str]     # {name: mcp_url}
    allowed_tools: List[str]         # Whitelist
    expected_impact: Optional[str] = None

class ExperimentConfig(BaseModel):
    experiment: BaseModel           # Metadata
    dataset: DatasetConfig
    execution: ExecutionConfig
    model: ModelConfig
    configs: List[PluginConfig]
    reporting: Dict
    pricing: Optional[Dict] = None

class ConfigManager:
    def __init__(self, yaml_path: str):
        self.config = self._load_and_validate(yaml_path)
    
    def _load_and_validate(self, path: str) -> ExperimentConfig:
        """Load YAML, parse, validate via Pydantic"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return ExperimentConfig(**data)
```

#### 4.2.3 Validation Rules

- All required fields present
- Numeric ranges valid (runs_per_instance >= 1, etc.)
- Model name is recognized Claude model
- MCP server URLs follow `mcp://` scheme
- Plugin IDs are unique within experiment

---

### 4.3 Runner Engine (`runner.py`)

**Responsibility**: Execute agent on each task and collect detailed metrics.

#### 4.3.1 Core Loop Architecture

```python
class BenchmarkRunner:
    def __init__(self, config: ExperimentConfig, dataset: List[SWEBenchInstance]):
        self.config = config
        self.dataset = dataset
        self.results = []  # Accumulated results
        
    async def run_all(self) -> BenchmarkResults:
        """
        Main execution entry point
        """
        for plugin_config in self.config.configs:
            results_for_config = await self._run_config(plugin_config)
            self.results.extend(results_for_config)
        return self._aggregate_results()
    
    async def _run_config(self, plugin_config: PluginConfig) -> List[RunRecord]:
        """
        For a single plugin config, run all instances N times
        """
        tasks = []
        for instance in self.dataset:
            for run_num in range(self.config.execution.runs_per_instance):
                task = asyncio.create_task(
                    self._run_single(instance, plugin_config, run_num)
                )
                tasks.append(task)
        
        # Respect max_parallel_tasks limit
        semaphore = asyncio.Semaphore(
            self.config.execution.max_parallel_tasks
        )
        
        async def bounded_run(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[bounded_run(t) for t in tasks])
        return results
    
    async def _run_single(
        self,
        instance: SWEBenchInstance,
        plugin_config: PluginConfig,
        run_num: int
    ) -> RunRecord:
        """
        Execute single instance and collect metrics
        """
        run_id = f"{instance.instance_id}_{plugin_config.id}_run{run_num}"
        
        try:
            # 1. Setup environment
            await self._setup_environment(instance)
            
            # 2. Create agent
            agent = await self._create_agent(plugin_config)
            
            # 3. Execute agent
            result = await self._execute_agent(agent, instance, run_id)
            
            # 4. Collect metrics
            record = RunRecord(
                run_id=run_id,
                instance_id=instance.instance_id,
                config_id=plugin_config.id,
                timestamp=datetime.now(),
                **result
            )
            
            return record
        
        except Exception as e:
            return RunRecord(
                run_id=run_id,
                instance_id=instance.instance_id,
                config_id=plugin_config.id,
                success=False,
                error_reason=str(e),
                duration_sec=0,
                tokens_input=0,
                tokens_output=0,
                tool_calls_total=0
            )
```

#### 4.3.2 Environment Setup

```python
class EnvironmentManager:
    """Manages Docker containers for SWE-bench instances"""
    
    async def setup(self, instance: SWEBenchInstance) -> str:
        """
        Prepares isolated environment:
        1. Clone repo at base_commit
        2. Install dependencies via setup.py / requirements.txt
        3. Verify tests run cleanly (no patches applied yet)
        
        Returns: Docker container ID or local directory path
        """
        container_id = f"swe_{instance.instance_id}_{uuid.uuid4().hex[:8]}"
        
        # Use SWE-bench official Docker setup
        subprocess.run([
            "docker", "run", "-d",
            "--name", container_id,
            f"swe-bench:env-{instance.repo_type}",
            f"bash -c 'cd /repo && git checkout {instance.base_commit}'"
        ])
        
        return container_id
    
    async def cleanup(self, container_id: str):
        """Stop and remove Docker container"""
        subprocess.run(["docker", "stop", container_id])
        subprocess.run(["docker", "rm", container_id])
```

#### 4.3.3 Claude Agent SDK Integration

**Key Challenge**: Extracting token/tool call data from streaming responses.

```python
class ClaudeAgentRunner:
    """Interface to Claude Agent SDK"""
    
    def __init__(self, model_config: ModelConfig, mcp_servers: Dict):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_config = model_config
        self.mcp_servers = mcp_servers
        self.metrics = MetricsCollector()
    
    async def execute(
        self,
        instance: SWEBenchInstance,
        allowed_tools: List[str]
    ) -> ExecutionResult:
        """
        Run agent until completion, termination, or timeout.
        
        Returns: ExecutionResult with metrics
        """
        
        # Prepare system prompt
        system_prompt = self._format_system_prompt(instance)
        
        # Prepare user message
        user_message = f"""
Please resolve this GitHub issue by modifying the source code.

**Issue Description:**
{instance.problem_statement}

**Expected Behavior:**
The test suite should pass with your modifications.

**Available Test Command:**
{instance.test_cmd}
"""
        
        messages = [{"role": "user", "content": user_message}]
        
        start_time = time.perf_counter()
        start_token_count = 0
        tool_calls_made = []
        
        try:
            # Call Claude with tools
            response = await self.client.messages.create(
                model=self.model_config.name,
                max_tokens=self.model_config.max_tokens,
                temperature=self.model_config.temperature,
                system=system_prompt,
                tools=self._build_tool_definitions(allowed_tools),
                messages=messages
            )
            
            # Extract token usage from response
            usage = response.usage
            total_input_tokens = usage.input_tokens
            total_output_tokens = usage.output_tokens
            
            # Process response content
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_calls_made.append({
                        "name": content_block.name,
                        "id": content_block.id,
                        "status": "pending"
                    })
                elif content_block.type == "text":
                    self.metrics.log_text_response(content_block.text)
            
            # Handle tool results in loop
            # (Agentic loop continues until stop_reason == "end_turn")
            while response.stop_reason == "tool_use":
                # Process tool results
                messages.append({"role": "assistant", "content": response.content})
                
                # Build tool results message
                tool_results = await self._execute_tools(tool_calls_made)
                messages.append({"role": "user", "content": tool_results})
                
                # Next iteration
                response = await self.client.messages.create(
                    model=self.model_config.name,
                    max_tokens=self.model_config.max_tokens,
                    temperature=self.model_config.temperature,
                    system=system_prompt,
                    tools=self._build_tool_definitions(allowed_tools),
                    messages=messages
                )
                
                # Accumulate token usage
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens
                
                # Track tool calls
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_calls_made.append({
                            "name": content_block.name,
                            "status": "executed"
                        })
            
            duration = time.perf_counter() - start_time
            
            return ExecutionResult(
                success=True,
                duration_sec=duration,
                tokens_input=total_input_tokens,
                tokens_output=total_output_tokens,
                tool_calls_total=len(tool_calls_made),
                tool_calls_by_name=self._count_by_name(tool_calls_made),
                final_message=response.content[-1].text if response.content else ""
            )
        
        except asyncio.TimeoutError:
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                success=False,
                error_reason="timeout",
                duration_sec=duration,
                tokens_input=0,
                tokens_output=0,
                tool_calls_total=len(tool_calls_made)
            )
        
        except Exception as e:
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                success=False,
                error_reason=str(e),
                duration_sec=duration,
                tokens_input=0,
                tokens_output=0
            )
    
    def _build_tool_definitions(self, allowed_tools: List[str]) -> List[Dict]:
        """
        Build Claude tool definitions based on allowed_tools whitelist.
        Filters to include only specified tools.
        """
        all_tools = {
            "read_file": {
                "name": "read_file",
                "description": "Read file contents",
                "input_schema": {...}
            },
            "write_file": {
                "name": "write_file",
                "description": "Write to file",
                "input_schema": {...}
            },
            "search_repo": {
                "name": "search_repo",
                "description": "Search repository code",
                "input_schema": {...}
            },
            # ... more tools
        }
        
        # Return only allowed tools
        return [
            tool for name, tool in all_tools.items()
            if name in allowed_tools or any(
                allowed.endswith("*") and name.startswith(allowed[:-1])
                for allowed in allowed_tools
            )
        ]
    
    async def _execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute requested tools and return results"""
        results = []
        for tool_call in tool_calls:
            try:
                result = await self._execute_single_tool(
                    tool_call["name"],
                    tool_call.get("input", {})
                )
                tool_call["status"] = "success"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": result
                })
            except Exception as e:
                tool_call["status"] = "failed"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": f"Error: {str(e)}",
                    "is_error": True
                })
        return results
```

#### 4.3.4 Metrics Collection

```python
@dataclass
class RunRecord:
    """Single benchmark run result"""
    run_id: str
    instance_id: str
    config_id: str
    timestamp: datetime
    
    # Outcome
    success: bool                          # Patch applied + tests passed
    error_reason: Optional[str] = None
    
    # Performance
    duration_sec: float
    
    # Efficiency
    tokens_input: int
    tokens_output: int
    tool_calls_total: int
    tool_calls_by_name: Dict[str, int]
    
    # Optional: cost approximation
    estimated_cost_usd: Optional[float] = None
```

---

### 4.4 Metrics & Aggregation (`metrics.py`)

**Responsibility**: Compute summary statistics and efficiency scores.

#### 4.4.1 Per-Configuration Aggregation

```python
@dataclass
class ConfigSummary:
    config_id: str
    config_name: str
    
    # Success metrics
    success_count: int
    total_runs: int
    success_rate: float
    
    # Timing (seconds)
    duration_mean: float
    duration_p50: float
    duration_p90: float
    duration_std: float
    
    # Tool calls (average per successful run)
    tool_calls_mean: float
    tool_calls_p90: float
    
    # Token usage
    tokens_input_mean: int
    tokens_output_mean: int
    tokens_total_mean: int
    
    # Cost (if configured)
    estimated_cost_per_run: Optional[float] = None
    total_experiment_cost: Optional[float] = None
    
    # Efficiency score (0-100, composite)
    efficiency_score: float
    
    def compute_efficiency_score(self, baseline_config: Optional['ConfigSummary'] = None):
        """
        Composite score: (Success Rate) / (Cost Normalized)
        
        Example:
        - Config A: 70% success, $2 per run → score = 70 / 2 = 35
        - Config B: 65% success, $1 per run → score = 65 / 1 = 65
        → Config B is more efficient (better rate per cost)
        """
        if not baseline_config:
            # Absolute score
            cost_per_run = self.estimated_cost_per_run or (
                (self.tokens_input_mean + self.tokens_output_mean) / 1_000_000 * 9  # Avg cost
            )
            self.efficiency_score = (self.success_rate * 100) / max(cost_per_run, 0.01)
        else:
            # Relative to baseline
            baseline_cost = baseline_config.estimated_cost_per_run or (
                (baseline_config.tokens_input_mean + baseline_config.tokens_output_mean) / 1_000_000 * 9
            )
            current_cost = self.estimated_cost_per_run or (
                (self.tokens_input_mean + self.tokens_output_mean) / 1_000_000 * 9
            )
            
            improvement = (self.success_rate - baseline_config.success_rate) * 100
            cost_delta = current_cost - baseline_cost
            
            # Improvement per dollar spent
            if cost_delta > 0:
                self.efficiency_score = improvement / cost_delta
            else:
                self.efficiency_score = improvement  # Cost savings = +bonus

class MetricsAggregator:
    def aggregate(self, records: List[RunRecord]) -> List[ConfigSummary]:
        """Group by config_id and compute statistics"""
        results_by_config = {}
        
        for record in records:
            if record.config_id not in results_by_config:
                results_by_config[record.config_id] = []
            results_by_config[record.config_id].append(record)
        
        summaries = []
        for config_id, runs in results_by_config.items():
            summary = ConfigSummary(
                config_id=config_id,
                config_name=runs[0].config_id,  # From RunRecord
                success_count=sum(1 for r in runs if r.success),
                total_runs=len(runs),
                success_rate=sum(1 for r in runs if r.success) / len(runs),
                
                duration_mean=statistics.mean([r.duration_sec for r in runs]),
                duration_p50=statistics.quantiles([r.duration_sec for r in runs], n=100)[49],
                duration_p90=statistics.quantiles([r.duration_sec for r in runs], n=100)[89],
                duration_std=statistics.stdev([r.duration_sec for r in runs]) if len(runs) > 1 else 0,
                
                tool_calls_mean=statistics.mean([r.tool_calls_total for r in runs if r.success]) or 0,
                tool_calls_p90=statistics.quantiles(
                    [r.tool_calls_total for r in runs if r.success],
                    n=100
                )[89] if any(r.success for r in runs) else 0,
                
                tokens_input_mean=int(statistics.mean([r.tokens_input for r in runs])),
                tokens_output_mean=int(statistics.mean([r.tokens_output for r in runs])),
                tokens_total_mean=int(statistics.mean([r.tokens_input + r.tokens_output for r in runs])),
            )
            
            # Compute efficiency score
            summary.compute_efficiency_score()
            
            summaries.append(summary)
        
        return summaries
```

#### 4.4.2 Export Formats

**JSON Export**:
```json
{
  "experiment": {
    "name": "swe-bench-plugin-ablation-v1",
    "created_at": "2024-12-03T21:00:00Z",
    "duration_total_minutes": 47
  },
  "configs": [
    {
      "config_id": "baseline_no_plugins",
      "config_name": "Baseline (No MCP Tools)",
      "success_rate": 0.32,
      "success_count": 16,
      "total_runs": 50,
      "duration_mean_sec": 38.5,
      "duration_p90_sec": 52.3,
      "tokens_input_mean": 3450,
      "tokens_output_mean": 1200,
      "tool_calls_mean": 0.0,
      "efficiency_score": 16.0,
      "estimated_cost_per_run_usd": 0.31
    },
    {
      "config_id": "with_file_editor",
      "config_name": "File Editor Only",
      "success_rate": 0.48,
      "success_count": 24,
      "total_runs": 50,
      "duration_mean_sec": 42.1,
      "duration_p90_sec": 58.7,
      "tokens_input_mean": 3890,
      "tokens_output_mean": 1450,
      "tool_calls_mean": 3.2,
      "efficiency_score": 24.8,
      "estimated_cost_per_run_usd": 0.38
    }
  ],
  "raw_runs": [
    {
      "run_id": "django__django-16379_with_file_editor_run0",
      "instance_id": "django__django-16379",
      "config_id": "with_file_editor",
      "success": true,
      "duration_sec": 41.2,
      "tokens_input": 3890,
      "tokens_output": 1450,
      "tool_calls_total": 4,
      "tool_calls_by_name": {"read_file": 3, "write_file": 1}
    }
  ]
}
```

---

### 4.5 TUI Display Module (`tui.py`)

**Responsibility**: Real-time terminal UI using Textual framework.

#### 4.5.1 Layout & Components

```
┌─────────────────────────────────────────────────────────────────┐
│ SWE-Bench Plugin Benchmark Tool                                │
├─────────────────────────────────────────────────────────────────┤
│ Experiment: swe-bench-plugin-ablation-v1                        │
│ Progress: [████████░░] 45/100 runs (45%)  ⏱ 23:45 elapsed      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Config Comparison:                                              │
│ ┌──────────────────┬────────┬───────┬───────┬───────────────┐   │
│ │ Config ID        │ Succes │ Time  │ Tools │ Cost/$        │   │
│ ├──────────────────┼────────┼───────┼───────┼───────────────┤   │
│ │ baseline         │ 30.0%  │ 38.5s │ 0.0   │ $0.31         │   │
│ │ with_file_editor │ 48.0%  │ 42.1s │ 3.2   │ $0.38         │   │
│ │ full_stack       │ 52.0%  │ 45.2s │ 7.8   │ $0.52         │   │
│ └──────────────────┴────────┴───────┴───────┴───────────────┘   │
│                                                                  │
│ Current Task:                                                   │
│ Instance: django__django-16379 [with_file_editor]              │
│ Status: Running tool call... (tool: search_repo)              │
│ Elapsed: 12.3s / Timeout: 900s                                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ 📊 Performance Distribution (Success Rate):                      │
│ baseline:      ██░░░░░░░░ 30%                                   │
│ with_editor:   ████░░░░░░ 48%                                   │
│ full_stack:    █████░░░░░ 52%                                   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ [q] Quit  [s] Save Report Now  [r] Reset  [l] View Logs       │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.5.2 Implementation (Textual)

```python
from textual.app import ComposeResult, App
from textual.widgets import DataTable, Header, Footer, ProgressBar
from textual.containers import Container

class BenchmarkApp(App):
    """Main TUI application"""
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #header {
        height: 3;
        background: $panel;
    }
    
    #progress_container {
        height: 4;
        border: solid $accent;
    }
    
    #comparison_table {
        height: 1fr;
    }
    
    #status_panel {
        height: 4;
        border: solid $accent;
    }
    
    #footer {
        height: 2;
        background: $panel;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container(id="progress_container"):
            yield ProgressBar(total=100, id="overall_progress")
        
        yield DataTable(id="comparison_table")
        
        with Container(id="status_panel"):
            yield Label("Current: Initializing...", id="status_label")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize tables and data"""
        table = self.query_one(DataTable)
        table.add_columns(
            "Config ID", "Success %", "Avg Time (s)", 
            "Tool Calls", "Tokens", "Cost ($)"
        )
        
        # Will be populated via update_results_table()
    
    def update_results_table(self, summaries: List[ConfigSummary]) -> None:
        """Refresh table with latest aggregated results"""
        table = self.query_one(DataTable)
        
        # Clear existing (if exists)
        while table.row_count > 0:
            table.remove_row(table.ordered_rows[0])
        
        for summary in summaries:
            table.add_row(
                summary.config_id,
                f"{summary.success_rate * 100:.1f}%",
                f"{summary.duration_mean:.1f}s",
                f"{summary.tool_calls_mean:.1f}",
                f"{summary.tokens_total_mean:,.0f}",
                f"${summary.estimated_cost_per_run_usd:.2f}" if summary.estimated_cost_per_run_usd else "N/A"
            )
    
    def update_status(self, current_task: str, elapsed_sec: int, timeout_sec: int) -> None:
        """Update current task status"""
        label = self.query_one(Label, id="status_label")
        label.update(f"Current: {current_task} | {elapsed_sec}s / {timeout_sec}s")
    
    def update_progress(self, completed: int, total: int) -> None:
        """Update overall progress bar"""
        progress = self.query_one(ProgressBar)
        progress.update(total=total, progress=completed)

class BenchmarkMonitor:
    """Bridges runner events to TUI updates"""
    
    def __init__(self, app: BenchmarkApp):
        self.app = app
    
    def on_results_aggregated(self, summaries: List[ConfigSummary]) -> None:
        """Called by runner when results are ready"""
        self.app.call_from_thread(
            lambda: self.app.update_results_table(summaries)
        )
    
    def on_task_status_changed(self, task: str, elapsed: int, timeout: int) -> None:
        """Called by runner during execution"""
        self.app.call_from_thread(
            lambda: self.app.update_status(task, elapsed, timeout)
        )
    
    def on_progress_updated(self, completed: int, total: int) -> None:
        """Called by runner on completion of each run"""
        self.app.call_from_thread(
            lambda: self.app.update_progress(completed, total)
        )
```

#### 4.5.3 User Interactions

| Key | Action |
| --- | --- |
| `q` | Quit (saves current results) |
| `s` | Save report immediately |
| `p` | Pause/resume execution |
| `l` | View detailed logs for selected config |
| `↑/↓` | Navigate table |
| `Enter` | Show detail view for selected config |

---

### 4.6 Report Generation (`reporter.py`)

**Responsibility**: Export results in multiple formats.

#### 4.6.1 Output Formats

**1. JSON** (`report.json`)
- Complete nested structure
- Machine-readable for post-analysis
- Includes raw per-run data

**2. YAML** (`report.yaml`)
- Human-readable
- Good for Git version control
- Summary statistics

**3. HTML** (`report.html`, optional)
- Interactive charts (chart.js or plotly)
- Comparison tables
- Export to PDF

#### 4.6.2 Report Structure

```python
class Reporter:
    def generate_all(
        self,
        results: BenchmarkResults,
        config: ExperimentConfig,
        output_dir: str
    ) -> Dict[str, str]:
        """Generate reports in all configured formats"""
        
        json_path = self._generate_json(results, config, output_dir)
        yaml_path = self._generate_yaml(results, config, output_dir)
        html_path = self._generate_html(results, config, output_dir)
        
        return {
            "json": json_path,
            "yaml": yaml_path,
            "html": html_path
        }
    
    def _generate_json(self, results, config, output_dir) -> str:
        """Export to JSON"""
        path = f"{output_dir}/report.json"
        with open(path, "w") as f:
            json.dump({
                "experiment": config.experiment.dict(),
                "summary": {
                    "total_runs": len(results.records),
                    "total_time_minutes": results.total_duration / 60,
                    "overall_success_rate": sum(1 for r in results.records if r.success) / len(results.records)
                },
                "configs": [s.dict() for s in results.summaries],
                "raw_runs": [r.dict() for r in results.records]
            }, f, indent=2)
        return path
    
    def _generate_html(self, results, config, output_dir) -> str:
        """Export to interactive HTML"""
        path = f"{output_dir}/report.html"
        
        # Use Jinja2 template
        html = self._render_template("report_template.html", {
            "experiment_name": config.experiment.name,
            "configs": results.summaries,
            "chart_data": self._prepare_chart_data(results),
            "generated_at": datetime.now().isoformat()
        })
        
        with open(path, "w") as f:
            f.write(html)
        
        return path
```

---

## 5. Technical Implementation Details

### 5.1 Claude Agent SDK Usage Patterns

#### 5.1.1 Cost/Token Tracking

The Claude Agent SDK provides token usage in response objects:

```python
# After each API call to Claude
response = await client.messages.create(...)

# Access token usage
input_tokens = response.usage.input_tokens
output_tokens = response.usage.output_tokens

# For multi-turn conversations, accumulate
total_input += input_tokens
total_output += output_tokens
```

**Key Point**: Each call to `.messages.create()` is a separate "step" in the agent loop. If the agent uses tools, you get N+1 API calls (1 initial + N tool result rounds).

#### 5.1.2 Tool Definition

```python
tools = [
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file (relative to repo root)"
                }
            },
            "required": ["file_path"]
        }
    },
    # More tools...
]

# Pass to Claude
response = await client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    tools=tools,  # Only include tools in allowed_tools list
    messages=[...]
)
```

#### 5.1.3 Tool Use Response Handling

```python
# Check response stop reason
if response.stop_reason == "tool_use":
    # Claude wants to call a tool
    tool_calls = [c for c in response.content if c.type == "tool_use"]
    
    for tool_call in tool_calls:
        # Execute the tool
        result = await execute_tool(tool_call.name, tool_call.input)
        
        # Add tool result to conversation
        messages.append({"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": result
        }]})

elif response.stop_reason == "end_turn":
    # Agent finished
    break

elif response.stop_reason == "max_tokens":
    # Ran out of tokens
    break
```

### 5.2 SWE-Bench Environment Setup

#### 5.2.1 Docker Integration

SWE-bench provides official Docker containers for reproducible evaluation:

```dockerfile
# Base image includes all dependencies
FROM swe-bench:base

# Environment-specific image (e.g., Python 3.9)
FROM swe-bench:env-python-39

# Instance-specific (contains cloned repo at base_commit)
FROM swe-bench:instance-django-16379
```

#### 5.2.2 Test Execution

After agent modifies code:

```bash
# 1. Apply patch (generated by Claude)
cd /repo
git apply agent_patch.diff

# 2. Run test command
bash -c "python -m pytest tests/path/to/test.py -xvs"

# 3. Capture exit code (0 = pass, non-zero = fail)
```

### 5.3 Reproducibility Mechanisms

#### 5.3.1 Random Seed Control

```python
# In config:
random_seed: 42

# In runner:
random.seed(config.random_seed)
np.random.seed(config.random_seed)
os.environ["PYTHONHASHSEED"] = str(config.random_seed)

# Claude temperature set to 0.2 (deterministic)
```

#### 5.3.2 Config Snapshot

Save used config to output directory:

```
results/
  ├─ report.json
  ├─ report.yaml
  ├─ config_used.yaml      # ← Snapshot of input config
  └─ logs/
     └─ raw_runs/
```

---

## 6. Deployment & Operations

### 6.1 System Requirements

| Component | Requirement |
| --- | --- |
| **OS** | Linux (recommended), macOS, Windows WSL2 |
| **Python** | 3.10+ |
| **RAM** | 16 GB (8 GB minimum) |
| **Disk** | 120 GB free (SWE-bench Docker images) |
| **Docker** | Docker Desktop or Docker Engine |
| **API** | ANTHROPIC_API_KEY environment variable |

### 6.2 Installation

```bash
# Clone repo
git clone https://github.com/yourusername/swe-bench-plugin-harness.git
cd swe-bench-plugin-harness

# Install dependencies
pip install -r requirements.txt

# Verify Docker
docker ps

# Set API key
export ANTHROPIC_API_KEY="sk-..."

# Run quick test
python benchmark.py --config examples/quick_test.yaml --dry-run
```

### 6.3 Usage

```bash
# Run full benchmark
python benchmark.py --config experiments.yaml

# Monitor TUI, results saved to ./results/report.{json,yaml,html}
```

---

## 7. Implementation Roadmap

### Phase 1: MVP (Weeks 1-2)
- ✅ YAML config parsing
- ✅ Dataset loading (HF)
- ✅ Single runner loop (Claude API calls, token tracking)
- ✅ JSON export

### Phase 2: Polish (Weeks 3-4)
- ✅ TUI display (Textual)
- ✅ Multi-config execution
- ✅ HTML report generation
- ✅ Error handling & logging

### Phase 3: Optimization (Week 5+)
- ✅ Parallel execution
- ✅ Caching optimizations
- ✅ Cost tracking
- ✅ Community documentation

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Claude API rate limits | Benchmark hangs | Implement exponential backoff + queue |
| Docker disk exhaustion | Benchmark fails | Pre-cleanup, cache level selection |
| Long test timeouts | Benchmark takes >8h | Configurable per-tool timeout, parallelization |
| Non-deterministic tests | Variance in results | Fixed seed, run N times, report std dev |
| Token overflow | Crashes on large contexts | Max_tokens setting + context compaction |

---

## 9. Appendix: File Structure

```
swe-bench-plugin-harness/
├── benchmark.py              # CLI entry point
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── dataset_loader.py     # HuggingFace integration
│   ├── runner.py             # Main execution engine
│   ├── claude_agent.py       # Claude SDK wrapper
│   ├── environment.py        # Docker/repo setup
│   ├── metrics.py            # Aggregation & stats
│   ├── reporter.py           # Report generation
│   └── tui.py                # Textual TUI
├── examples/
│   ├── quick_test.yaml       # 5 instances, 2 configs
│   └── full_ablation.yaml    # 50 instances, 5 configs
├── templates/
│   └── report_template.html  # Jinja2 template
├── tests/
│   ├── test_config.py
│   ├── test_runner.py
│   └── test_metrics.py
├── docs/
│   ├── README.md
│   ├── SETUP.md
│   ├── CONFIG_GUIDE.md
│   └── API.md
└── .github/
    └── workflows/
        └── ci.yaml           # GitHub Actions
```

---

## 10. References & Citations

- [SWE-bench Paper (ACL 2024)](https://arxiv.org/abs/2310.06770)
- [Claude Agent SDK Docs](https://platform.claude.com/docs/en/agent-sdk)
- [SWE-bench Evaluation Guide](https://www.swebench.com/)
- [Textual Framework](https://textual.textualize.io/)
- [Anthropic Cost Tracking](https://platform.claude.com/docs/en/agent-sdk/cost-tracking)
- [MLflow Anthropic Integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/anthropic/)

---

**Document Version**: 1.0.0-design  
**Last Updated**: December 3, 2024  
**Status**: Ready for Implementation Review