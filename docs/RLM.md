# Recursive Language Models (RLM) in Synalinks

## Overview

Recursive Language Models (RLM) represent a paradigm shift in how LLMs handle long-context tasks. Instead of feeding entire documents into a fixed context window, RLM enables LLMs to **programmatically explore** large inputs by writing and executing Python code in an iterative REPL (Read-Eval-Print-Loop) environment.

This document covers the theory behind RLM, its implementation in Synalinks via the `RLM` class, and practical examples to get you started.

## Table of Contents

1. [The RLM Paper](#the-rlm-paper)
2. [How RLM Works](#how-rlm-works)
3. [Use Cases](#use-cases)
4. [Synalinks Implementation](#synalinks-implementation)
5. [Acknowledgments](#acknowledgments)
6. [Examples](#examples)
7. [API Reference](#api-reference)

---

## The RLM Paper

RLM was introduced in the paper **"Recursive Language Models"** by Alex L. Zhang, Tim Kraska, and Omar Khattab (December 2025).

**Paper:** [https://arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601)

### The Problem

Large Language Models have a fundamental limitation: **fixed context windows**. Even with models supporting 128K or 1M tokens, there are practical limits to how much context can be processed effectively. Traditional approaches include:

- **Chunking**: Split documents and process independently (loses cross-chunk context)
- **Summarization**: Compress content (loses detail)
- **RAG**: Retrieve relevant chunks (may miss connections)

These approaches all sacrifice information or context coherence.

### The RLM Solution

RLM proposes a fundamentally different approach: treat the long context as an **external environment** that the LLM can systematically navigate through code execution.

Key insight: Instead of asking "How do I fit this into my context window?", RLM asks "How can I programmatically explore this data to find what I need?"

### Key Results

The paper demonstrates:

- Handles inputs **100x longer** than native context windows
- **Outperforms** base LLMs on long-context tasks
- **Beats** common long-context scaffolding approaches
- Maintains **comparable or reduced cost** per query

---

## How RLM Works

### The REPL Loop

RLM operates through an iterative cycle:

```
1. LLM receives task + available variables
2. LLM writes Python code to explore/analyze data
3. Code executes in sandboxed interpreter
4. LLM sees output, decides next action
5. Repeat until LLM calls SUBMIT() with final answer
```

### Key Components

#### 1. Code Interpreter
A sandboxed Python environment where LLM-generated code executes safely. Variables persist across iterations, allowing the LLM to build up state incrementally.

#### 2. Sub-LLM Queries
The `llm_query(prompt)` function allows the LLM to make semantic queries about specific data portions. This enables hybrid reasoning: code for structure, LLM for semantics.

#### 3. SUBMIT Function
When the LLM has gathered enough information, it calls `SUBMIT(field1=value1, ...)` to produce the final structured output.

### Example Execution Flow

```
Iteration 0:
  Reasoning: "Let me first understand the document structure"
  Code: print(len(document)); print(document[:500])
  Output: "12450\n# Annual Report 2024\n\n## Executive Summary..."

Iteration 1:
  Reasoning: "I see sections. Let me find revenue information"
  Code: for line in document.split('\n'):
            if 'revenue' in line.lower():
                print(line)
  Output: "Total Revenue: $45.2 million (up 23% YoY)"

Iteration 2:
  Reasoning: "Found the answer, submitting"
  Code: SUBMIT(answer="Revenue was $45.2M, grew 23%", confidence=0.95)
```

---

## Use Cases

### 1. Document Analysis
Analyze long documents (reports, contracts, research papers) by programmatically navigating sections and extracting specific information.

### 2. Data Exploration
Explore complex JSON/structured data by writing code to traverse, filter, and aggregate.

### 3. Multi-Step Computation
Perform calculations that require multiple steps, where each step's output informs the next.

### 4. Knowledge Graph Construction
Extract entities and relationships from text by iteratively identifying and connecting concepts.

### 5. Code Understanding
Analyze codebases by programmatically traversing files, finding patterns, and understanding architecture.

### 6. Structured Extraction
Extract typed, structured data from unstructured text with validation and type coercion.

### 7. Business Intelligence
Query databases and APIs through custom tools, combining data from multiple sources.

---

## Synalinks Implementation

Synalinks implements RLM through the `RLM` class (with `REPLModule` as a backward-compatible alias), providing a clean, Keras-inspired API that integrates with the Synalinks ecosystem.

### Architecture

```
RLM
    |
    +-- REPLGenerator (generates code + reasoning)
    |       |
    |       +-- Uses output schema to build SUBMIT instructions
    |
    +-- CodeInterpreter (executes code safely)
    |       |
    |       +-- NativePythonInterpreter (default, restricted builtins)
    |       +-- DenoInterpreter (secure WASM sandbox via Deno/Pyodide)
    |
    +-- Tools (optional custom functions)
    |
    +-- REPLHistory (tracks execution trajectory)
```

### Core Classes

#### RLM
The main orchestrator that runs the REPL loop.

```python
import synalinks

rlm = synalinks.RLM(
    data_model=OutputModel,      # Pydantic model for output
    language_model=lm,           # LLM for code generation
    max_iterations=20,           # Max REPL iterations
    max_llm_calls=50,            # Max sub-LLM queries
    tools=[...],                 # Optional custom tools
    return_history=True,         # Include execution trajectory
    instructions="...",          # Custom instructions
)

# REPLModule is still available as a backward-compatible alias
# repl = synalinks.REPLModule(...)  # Works but deprecated
```

#### REPLGenerator
Specialized Generator that produces `REPLAction` outputs (reasoning + code).

#### Code Interpreters

Synalinks provides two interpreter implementations:

**NativePythonInterpreter** (Default)
- Runs in-process using Python's `exec()`
- Restricted builtins (no `__import__`, file I/O blocked)
- Variable persistence across iterations
- Fast startup, suitable for trusted environments
- Note: Limited security - code runs in the host Python process

**DenoInterpreter** (Secure Sandbox)
- Runs Python in a WebAssembly sandbox via Deno + Pyodide
- Complete isolation from host filesystem and network
- JSON-RPC 2.0 communication protocol
- Requires Deno installation: `brew install deno` or `curl -fsSL https://deno.land/install.sh | sh`
- Inspired by DSPy's PythonInterpreter implementation
- Recommended for untrusted code or production use

```python
from synalinks.interpreters import DenoInterpreter

# Use secure sandbox
rlm = synalinks.RLM(
    data_model=OutputModel,
    language_model=lm,
    interpreter=DenoInterpreter(),  # Secure WASM sandbox
)

# DenoInterpreter with custom permissions
interpreter = DenoInterpreter(
    enable_read_paths=["/data/inputs"],   # Allow reading specific paths
    enable_write_paths=["/data/outputs"], # Allow writing to specific paths
    enable_env_vars=["API_KEY"],          # Expose specific env vars
    enable_network_access=["api.example.com"],  # Allow network to specific hosts
)
```

#### REPLEntry / REPLHistory
Data models for tracking execution trajectory, useful for debugging and optimization.

### Comparison with DSPy

| Aspect | DSPy RLM | Synalinks RLM |
|--------|----------|----------------------|
| Schema Definition | String signatures (`"a, b -> c"`) | Pydantic DataModels |
| Module System | DSPy modules | Keras-style modules |
| Interpreter | PythonInterpreter (Deno/WASM) | NativePythonInterpreter or DenoInterpreter |
| Training | DSPy optimizers | Synalinks optimizers (OMEGA, etc.) |
| Tools | Dict of callables | List of Tool objects |

### Security Considerations

For production use with untrusted input, always use `DenoInterpreter`:

```python
# Production setup with secure sandbox
interpreter = DenoInterpreter(
    max_output_chars=50_000,  # Limit output size
)

rlm = synalinks.RLM(
    data_model=OutputModel,
    language_model=lm,
    interpreter=interpreter,
    max_iterations=10,  # Limit iterations
)
```

The DenoInterpreter provides:
- **Process isolation**: Python runs in a separate Deno subprocess
- **WASM sandboxing**: Pyodide runs Python compiled to WebAssembly
- **Virtual filesystem**: No access to host files by default
- **Controlled permissions**: Fine-grained `--allow-read`, `--allow-write`, `--allow-net` flags

---

## Acknowledgments

The Synalinks `RLM` implementation is **directly inspired by DSPy's RLM module**. We gratefully acknowledge:

- **DSPy Team** for pioneering the RLM approach and providing an excellent reference implementation
- **Alex L. Zhang, Tim Kraska, and Omar Khattab** for the foundational RLM research
- The open-source community for making this kind of cross-pollination possible

DSPy's RLM implementation served as the blueprint for understanding:
- The REPL loop structure
- Code generation prompting strategies
- Type coercion patterns
- Fallback extraction mechanisms

**DSPy Repository:** [https://github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)

---

## Examples

All examples are located in the `examples/` directory and can be run with:

```bash
cd ~/workspace/synalinks
uv run --env-file .env -- python examples/<example_name>.py
```

**Requirements:**
- `GROQ_API_KEY` environment variable set in `.env` file
- Synalinks installed (`uv sync`)

### Example 1: Basic Document Analysis

**File:** `examples/repl_module_example.py`

Demonstrates the fundamental RLM pattern: analyzing a document to answer a question.

```python
class DocumentQuery(synalinks.DataModel):
    document: str = synalinks.Field(description="The document to analyze")
    question: str = synalinks.Field(description="Question to answer")

class AnalysisResult(synalinks.DataModel):
    answer: str = synalinks.Field(description="Answer to the question")
    evidence: list = synalinks.Field(description="Supporting evidence")
    confidence: float = synalinks.Field(description="Confidence score 0-1")

rlm = synalinks.RLM(
    data_model=AnalysisResult,
    language_model=lm,
    max_iterations=10,
    return_history=True,
)
```

**Run:**
```bash
uv run --env-file .env -- python examples/repl_module_example.py
```

**Sample Output:**
```
ANALYSIS RESULT
Answer: The company's revenue was $45.2 million and it grew by 23%
Evidence: ['Revenue increased by 23% compared to the previous year...']
Confidence: 0.95
```

---

### Example 2: Document Analysis with Trajectory

**File:** `examples/rlm_document_analysis.py`

Extended document analysis showing the full execution trajectory, demonstrating how the LLM explores the document step by step.

**Key Features:**
- Shows reasoning at each iteration
- Displays code executed
- Shows intermediate outputs
- Demonstrates iterative exploration

**Run:**
```bash
uv run --env-file .env -- python examples/rlm_document_analysis.py
```

---

### Example 3: Data Analysis with Multi-Step Computation

**File:** `examples/rlm_data_analysis.py`

Demonstrates RLM for statistical analysis over structured data, requiring multiple computational steps.

```python
class DataAnalysisInput(synalinks.DataModel):
    data: dict = synalinks.Field(description="Structured data to analyze")
    question: str = synalinks.Field(description="Analysis question")

class AnalysisOutput(synalinks.DataModel):
    result: float = synalinks.Field(description="Numeric result")
    methodology: str = synalinks.Field(description="How computed")
    insights: str = synalinks.Field(description="Key insights")
```

**Example Questions:**
- "Calculate total profit and profit margin percentage"
- "Which region has the highest revenue per customer?"
- "Calculate weighted average profit margin across products"

**Run:**
```bash
uv run --env-file .env -- python examples/rlm_data_analysis.py
```

**Sample Output:**
```
ANALYSIS 1: Calculate total profit across all quarters
Result: 180000
Methodology: Summed quarterly profits (revenue - costs)
Insights: Q4 most profitable at $63k, 28.57% overall margin
[Completed in 3 iterations]
```

---

### Example 4: Custom Tools for Business Intelligence

**File:** `examples/rlm_custom_tools.py`

Shows how to extend RLM with custom tools that the LLM can call from within the REPL.

```python
@synalinks.saving.register_synalinks_serializable()
async def search_users(query: str) -> dict:
    """Search users by name, email, or plan."""
    # ... implementation
    return {"results": [...], "count": N}

@synalinks.saving.register_synalinks_serializable()
async def calculate_mrr() -> dict:
    """Calculate Monthly Recurring Revenue."""
    return {"total_mrr": 1999.94, "by_plan": {...}}

tools = [
    synalinks.Tool(search_users),
    synalinks.Tool(calculate_mrr),
]

rlm = synalinks.RLM(
    data_model=BusinessInsight,
    language_model=lm,
    tools=tools,
)
```

**Key Pattern:** Tools must be:
1. Async functions
2. Registered with `@register_synalinks_serializable()`
3. Have docstrings with Args descriptions
4. Return dicts (not strings)

**Run:**
```bash
uv run --env-file .env -- python examples/rlm_custom_tools.py
```

---

### Example 5: Structured Data Extraction

**File:** `examples/rlm_structured_extraction.py`

Demonstrates extracting typed, structured data from unstructured text.

```python
class CompanyProfile(synalinks.DataModel):
    company_name: str = synalinks.Field(description="Company name")
    industry: str = synalinks.Field(description="Industry sector")
    founded_year: int = synalinks.Field(description="Year founded")
    headquarters: str = synalinks.Field(description="HQ location")
    employee_count: int = synalinks.Field(description="Number of employees")
    annual_revenue: float = synalinks.Field(description="Revenue in millions")
    key_products: list = synalinks.Field(description="Main products")
    competitors: list = synalinks.Field(description="Main competitors")

class JobPosting(synalinks.DataModel):
    job_title: str = synalinks.Field(description="Position title")
    salary_min: float = synalinks.Field(description="Min salary in thousands")
    salary_max: float = synalinks.Field(description="Max salary in thousands")
    skills: list = synalinks.Field(description="Required skills")
    remote_friendly: bool = synalinks.Field(description="Remote allowed")
```

**Run:**
```bash
uv run --env-file .env -- python examples/rlm_structured_extraction.py
```

---

## API Reference

### RLM

```python
class RLM(Module):
    def __init__(
        self,
        schema: Optional[dict] = None,           # JSON schema for output
        data_model: Optional[Type] = None,       # Pydantic model (preferred)
        language_model=None,                      # Required: LLM for code gen
        sub_language_model=None,                  # Optional: cheaper LLM for queries
        interpreter: Optional[CodeInterpreter] = None,  # Default: NativePythonInterpreter
        tools: Optional[List[Tool]] = None,       # Custom tools
        max_iterations: int = 20,                 # Max REPL iterations
        max_llm_calls: int = 50,                  # Max sub-LLM queries
        max_output_chars: int = 100_000,          # Max output characters
        instructions: Optional[str] = None,       # Custom instructions
        seed_instructions: Optional[List[str]] = None,  # For optimization
        return_history: bool = False,             # Include trajectory
        name: Optional[str] = None,
        description: Optional[str] = None,
        trainable: bool = True,
    )
```

### Built-in REPL Functions

Available to the LLM within the REPL:

| Function | Description |
|----------|-------------|
| `print(...)` | Print output (ALWAYS use to see results) |
| `llm_query(prompt)` | Query sub-LLM for semantic analysis |
| `llm_query_batched(prompts)` | Batch query multiple prompts |
| `SUBMIT(field=value, ...)` | Submit final answer |

Plus all custom tools registered via the `tools` parameter.

### NativePythonInterpreter

```python
class NativePythonInterpreter(CodeInterpreter):
    def __init__(
        self,
        max_output_chars: int = 100_000,  # Truncate output beyond this
    )

    async def execute(
        self,
        code: str,
        variables: Optional[Dict] = None,
        tools: Optional[Dict] = None,
    ) -> str | FinalOutput
```

### REPLEntry / REPLHistory

```python
class REPLEntry(DataModel):
    iteration: int       # 0-indexed iteration number
    reasoning: str       # LLM's reasoning
    code: str           # Executed code
    stdout: str         # Captured output
    error: Optional[str] # Error if any

class REPLHistory(DataModel):
    entries: List[REPLEntry]

    def format_for_prompt(self) -> str  # Format for LLM context
    def had_errors(self) -> bool        # Check for errors
    def get_last_entry(self) -> REPLEntry
```

---

## Best Practices

### 1. Clear Instructions
Provide specific instructions about available variables and expected output format.

### 2. Appropriate Iteration Limits
Start with `max_iterations=10-20`. Increase for complex tasks.

### 3. Use Type Hints
Define output schemas with proper types for automatic coercion.

### 4. Enable History for Debugging
Use `return_history=True` during development to understand LLM behavior.

### 5. Custom Tools for Domain Logic
Encapsulate complex operations in tools rather than expecting LLM to write them.

---

## Troubleshooting

### LLM Doesn't SUBMIT
- Increase `max_iterations`
- Make instructions clearer about when to SUBMIT
- Check if output schema is too complex

### Type Coercion Errors
- Ensure schema types match expected LLM output
- Use string types for ambiguous fields
- Check for None values in required fields

### Tool Execution Errors
- Tools must be async functions
- Tools must return dicts, not strings
- Ensure proper docstrings with Args

---

## Future Work

- Docker-based interpreter for full Python support
- WASM interpreter for browser environments
- Training/optimization of REPL instructions
- Parallel REPL execution for independent subtasks

---

## References

1. Zhang, A. L., Kraska, T., & Khattab, O. (2025). Recursive Language Models. arXiv:2512.24601
2. DSPy: Programming with Foundation Models. https://github.com/stanfordnlp/dspy
3. Synalinks: Keras-inspired LLM Programming. https://github.com/synalinks/synalinks
