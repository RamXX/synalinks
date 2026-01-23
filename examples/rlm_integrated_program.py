#!/usr/bin/env python
# License Apache 2.0: (c) 2025 Synalinks Team

"""Integrated RLM Program - Deep Synalinks Documentation Analysis

This example demonstrates the full power of RLM (Recursive Language Model)
integrated into a Synalinks Program. It performs deep, multi-stage analysis
of Synalinks' own codebase using:

1. **Iterative Code Analysis** - RLM writes Python to parse source files
2. **Semantic Understanding** - llm_query() for understanding code semantics
3. **Cross-Reference Analysis** - Finding relationships between components
4. **Multi-Pass Refinement** - Iterating to build comprehensive understanding

This is dogfooding at scale: using Synalinks' most powerful features to
deeply understand Synalinks itself.

Architecture:
    Input (SourceFiles)
        |
        v
    RLM                     <-- Multi-iteration analysis with:
    (deep_analyzer)             - Code parsing via Python
        |                       - Semantic queries via llm_query()
        |                       - Cross-reference analysis
        v
    DeepAnalysis
        |
        +-- merge with inputs (&)
        |
        v
    Generator               <-- TRAINABLE synthesis
    (synopsis_writer)
        |
        v
    ComprehensiveSynopsis

Based on DSPy's RLM implementation:
    https://github.com/stanfordnlp/dspy

Requirements:
    - Set GROQ_API_KEY environment variable

Usage:
    uv run --env-file .env python examples/rlm_integrated_program.py
"""

import ast
import asyncio
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import field_validator

import synalinks
from synalinks.src.modules.reasoning import RLM


# =============================================================================
# Data Models - Rich schemas for deep analysis
# =============================================================================


class SourceFiles(synalinks.DataModel):
    """Input: Source files to analyze."""

    files: List[str] = synalinks.Field(
        description="List of file contents (code or documentation)"
    )
    file_names: List[str] = synalinks.Field(
        description="Names of the files being analyzed"
    )


class DeepAnalysis(synalinks.DataModel):
    """Output from RLM: Comprehensive multi-stage analysis."""

    # Stage 1: Structural discovery
    classes_found: List[str] = synalinks.Field(
        description="All class names discovered with their purpose"
    )
    key_patterns: List[str] = synalinks.Field(
        description="Recurring code patterns identified"
    )

    # Stage 2: Semantic understanding
    core_abstractions: List[str] = synalinks.Field(
        description="Core abstractions and what problems they solve"
    )
    design_principles: List[str] = synalinks.Field(
        description="Design principles evident in the codebase"
    )

    # Stage 3: Relationships
    module_relationships: List[str] = synalinks.Field(
        description="How different modules relate to each other"
    )
    data_flow: str = synalinks.Field(
        description="How data flows through a typical program"
    )

    # Stage 4: Unique insights
    novel_techniques: List[str] = synalinks.Field(
        description="Novel or unusual techniques used"
    )
    comparison_to_alternatives: str = synalinks.Field(
        description="How this compares to DSPy, LangChain, etc."
    )


class ComprehensiveSynopsis(synalinks.DataModel):
    """Final output: Deep synopsis of the framework."""

    elevator_pitch: str = synalinks.Field(
        description="30-second explanation for a developer"
    )

    architecture_overview: str = synalinks.Field(
        description="How the framework is architected (3-4 sentences)"
    )

    five_key_concepts: List[str] = synalinks.Field(
        description=(
            "The 5 most important concepts with brief explanations. "
            "Format each item as 'Concept — short explanation'."
        )
    )

    getting_started_code: str = synalinks.Field(
        description=(
            "Minimal but complete code example to get started. "
            "Must be valid runnable Python (no markdown fences, no ellipses, "
            "no triple quotes). Include imports, async main(), await Generator call, "
            "and asyncio.run(main()). Use ASCII quotes only."
        )
    )

    power_features: List[str] = synalinks.Field(
        description="Advanced features that set this apart"
    )

    when_to_use: str = synalinks.Field(
        description="When to choose this framework over alternatives"
    )

    @field_validator("getting_started_code")
    @classmethod
    def validate_getting_started_code(cls, value: str) -> str:
        try:
            ast.parse(value)
        except SyntaxError as exc:
            raise ValueError("getting_started_code must be valid Python") from exc

        required_snippets = ("async def main", "asyncio.run", "await ")
        if any(snippet not in value for snippet in required_snippets):
            raise ValueError(
                "getting_started_code must include async main(), await, and asyncio.run()"
            )
        return value


# =============================================================================
# Load Source Files
# =============================================================================

# Files to analyze (relative to repo root)
SOURCE_FILES = [
    "README.md",
    "docs/CheatSheet.md",
    "examples/1a_functional_api.py",
    "examples/4_conditional_branches.py",
    "examples/8_training_programs.py",
    "examples/10_autonomous_agent.py",
]

def load_sources(repo_root: Path, file_paths: list[str]) -> tuple[list[str], list[str]]:
    """Load source files for analysis.

    Note: No truncation needed - RLM passes data by reference to the REPL
    environment, NOT into the LLM's context window. The LLM writes code
    to explore the data programmatically.
    """
    contents = []
    names = []

    for file_path in file_paths:
        full_path = repo_root / file_path
        if full_path.exists():
            contents.append(full_path.read_text())
            names.append(file_path)

    return contents, names


# =============================================================================
# Build Integrated Program
# =============================================================================

GETTING_STARTED_TEMPLATE = """import asyncio
import synalinks

class Query(synalinks.DataModel):
    query: str

class Answer(synalinks.DataModel):
    answer: str

async def main():
    lm = synalinks.LanguageModel(model="gpt-4o-mini")
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs)
    result = await program(Query(query="Capital of France?"))
    print(result["answer"])

if __name__ == "__main__":
    asyncio.run(main())
"""


async def build_deep_analysis_program(code_lm, query_lm, synthesis_lm):
    """Build a program for deep multi-stage analysis.

    The RLM performs iterative analysis:
    1. Parse code to find classes, patterns, structure
    2. Use llm_query() to understand semantics
    3. Cross-reference to find relationships
    4. Synthesize insights

    Args:
        code_lm: Language model for RLM code generation
        query_lm: Language model for llm_query() semantic analysis
        synthesis_lm: Language model for final synthesis
    """
    inputs = synalinks.Input(data_model=SourceFiles)

    # RLM with extensive iteration for deep analysis
    # Note: Instructions follow DSPy's exploratory style - let the LLM discover
    analysis = await RLM(
        data_model=DeepAnalysis,
        language_model=code_lm,
        sub_language_model=query_lm,  # For llm_query() calls
        max_iterations=15,
        max_llm_calls=30,
        return_history=True,  # Enable history for introspection
        instructions="""Analyze the framework codebase to understand its architecture, design, and differentiators.""",
        name="deep_analyzer",
    )(inputs)

    # Merge for full context
    synthesis_context = inputs & analysis

    # Generator for comprehensive synopsis
    outputs = await synalinks.Generator(
        data_model=ComprehensiveSynopsis,
        language_model=synthesis_lm,
        instructions=(
            "Synthesize the analysis into a comprehensive framework synopsis. "
            "Be specific and technical, using actual concepts discovered in the codebase. "
            "For `five_key_concepts`, each item must include a short explanation. "
            "For `getting_started_code`, output valid runnable Python code only "
            "(no markdown, no ellipses, no triple quotes). Use ASCII quotes only. "
            "Follow this exact template:\n"
            f"{GETTING_STARTED_TEMPLATE.strip()}"
        ),
        name="synopsis_writer",
    )(synthesis_context)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="deep_synalinks_analyzer",
        description="Deep multi-stage analysis of Synalinks codebase",
    )

    return program


# =============================================================================
# Main
# =============================================================================


async def main():
    load_dotenv()

    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        return

    print("=" * 70)
    print("DEEP SYNALINKS ANALYSIS")
    print("Multi-Stage Recursive Analysis Using RLM")
    print("=" * 70)
    print("""
This example demonstrates the full power of RLM integration:

    [8 Source Files]
           |
           v
    [RLM]         -----> Iteration 1: Parse classes & patterns
           |      -----> Iteration 2: llm_query() for semantics
           |      -----> Iteration 3: Cross-reference analysis
           |      -----> Iteration 4+: Refine & synthesize
           v
    [DeepAnalysis]
           |
           +-- merge (&)
           |
           v
    [Generator]
           |
           v
    [ComprehensiveSynopsis]

The RLM iterates multiple times, using:
- Python code for structural analysis
- llm_query() for semantic understanding
- Cross-referencing for relationship mapping
""")

    # Initialize language models
    # Main model for code generation (strong coding model)
    code_lm = synalinks.LanguageModel(model="groq/moonshotai/kimi-k2-instruct-0905", timeout=120)
    # Sub-model for llm_query() calls (fast, efficient)
    query_lm = synalinks.LanguageModel(model="groq/openai/gpt-oss-20b", timeout=60)
    # Synthesis model
    synthesis_lm = synalinks.LanguageModel(model="groq/moonshotai/kimi-k2-instruct-0905", timeout=90)

    # -------------------------------------------------------------------------
    # Build the program
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("Building deep analysis program...")
    print("-" * 70)

    program = await build_deep_analysis_program(code_lm, query_lm, synthesis_lm)

    print("\nProgram structure:")
    program.summary()

    # Generate visualization
    synalinks.utils.plot_program(
        program,
        to_folder="examples",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )
    print("\nProgram diagram: examples/deep_synalinks_analyzer.png")

    # -------------------------------------------------------------------------
    # Load source files for analysis
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Loading source files...")
    print("-" * 70)

    repo_root = Path(__file__).parent.parent
    file_contents, file_names = load_sources(repo_root, SOURCE_FILES)

    print(f"Loaded {len(file_names)} files:")
    for name in file_names:
        print(f"  - {name}")

    total_chars = sum(len(f) for f in file_contents)
    print(f"\nTotal content: {total_chars:,} characters")

    test_input = SourceFiles(
        files=file_contents,
        file_names=file_names,
    )

    # -------------------------------------------------------------------------
    # Run deep analysis
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Running deep multi-stage analysis...")
    print("(RLM will iterate: parse -> query -> cross-reference -> refine)")
    print("-" * 70)

    result = await program(test_input)

    if result:
        print("\n" + "=" * 70)
        print("COMPREHENSIVE SYNALINKS SYNOPSIS")
        print("=" * 70)

        print(f"\n{'='*70}")
        print("ELEVATOR PITCH")
        print("="*70)
        print(result['elevator_pitch'])

        print(f"\n{'='*70}")
        print("ARCHITECTURE OVERVIEW")
        print("="*70)
        print(result['architecture_overview'])

        print(f"\n{'='*70}")
        print("FIVE KEY CONCEPTS")
        print("="*70)
        for i, concept in enumerate(result.get('five_key_concepts', []), 1):
            print(f"\n{i}. {concept}")

        print(f"\n{'='*70}")
        print("GETTING STARTED CODE")
        print("="*70)
        print(result['getting_started_code'])

        print(f"\n{'='*70}")
        print("POWER FEATURES")
        print("="*70)
        for feature in result.get('power_features', []):
            print(f"  - {feature}")

        print(f"\n{'='*70}")
        print("WHEN TO USE SYNALINKS")
        print("="*70)
        print(result['when_to_use'])

        print("\n" + "=" * 70)
        print("END OF PROGRAM OUTPUT")
        print("=" * 70)
        print("The above synopsis was generated by analyzing Synalinks' own codebase.")
        print("What follows is execution metadata and statistics.")

    # -------------------------------------------------------------------------
    # Execution Statistics (RLM Introspection)
    # -------------------------------------------------------------------------
    # Note: We show statistics regardless of result success for debugging
    print("\n" + "=" * 70)
    print("EXECUTION STATISTICS")
    print("=" * 70)

    if not result:
        print("\nWARNING: Program returned no result (all LLM calls may have failed)")
        print("Check API keys and rate limits.")

    # Extract history from result (stored as _history when return_history=True)
    history = result.get('_history', []) if result else []

    # Note: In this integrated example, the _history from RLM gets lost
    # when passing through the synthesis Generator. For full trajectory access,
    # either use RLM as the final output or extend the output schema.
    if history:
        print(f"\nREPL Iterations: {len(history)}")

        # Count llm_query calls by scanning code in history
        llm_query_calls = sum(
            entry.get('code', '').count('llm_query(') +
            entry.get('code', '').count('llm_query_batched(')
            for entry in history
        )
        print(f"Sub-LLM Calls (llm_query): {llm_query_calls}")

        # Count errors (check 'error' field which contains the error message)
        error_count = sum(
            1 for entry in history
            if entry.get('error') is not None
        )
        print(f"Errors Encountered: {error_count}")

        # Show condensed trajectory
        print(f"\n{'='*70}")
        print("REPL TRAJECTORY (Condensed)")
        print("="*70)
        for entry in history:
            iteration = entry.get('iteration', 0) + 1
            reasoning = entry.get('reasoning', '')[:80]
            if len(entry.get('reasoning', '')) > 80:
                reasoning += "..."
            code_lines = len(entry.get('code', '').strip().split('\n'))

            # Show stdout or error
            stdout = entry.get('stdout', '') or ''
            error = entry.get('error')
            if error:
                output_preview = f"ERROR: {str(error)[:50]}"
            else:
                output_preview = stdout[:60].replace('\n', ' ')
                if len(stdout) > 60:
                    output_preview += "..."

            print(f"\n[Iteration {iteration}]")
            print(f"  Reasoning: {reasoning}")
            print(f"  Code: {code_lines} lines")
            print(f"  Output: {output_preview}")
    else:
        print("\nREPL history not in final output (lost in Generator synthesis step)")
        print("The RLM did execute - see LM costs above for evidence.")
        print("\nTo preserve trajectory in production:")
        print("  1. Use synalinks.callbacks for telemetry (MLflow, custom loggers)")
        print("  2. Structure program to expose intermediate RLM outputs")
        print("  3. Add _history field to final output schema as pass-through")

    # Language model costs (if available)
    print(f"\n{'='*70}")
    print("LANGUAGE MODEL COSTS")
    print("="*70)
    print(f"  Code LM (code_lm):      ${code_lm.cumulated_cost:.6f}")
    print(f"  Query LM (query_lm):    ${query_lm.cumulated_cost:.6f}")
    print(f"  Synthesis LM:           ${synthesis_lm.cumulated_cost:.6f}")
    total_cost = code_lm.cumulated_cost + query_lm.cumulated_cost + synthesis_lm.cumulated_cost
    print(f"  ----------------------------------------")
    print(f"  TOTAL:                  ${total_cost:.6f}")

    # -------------------------------------------------------------------------
    # Program capabilities
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Program capabilities:")
    print("-" * 70)
    print(f"Total modules: {len(program.modules)}")
    print(f"Trainable variables: {len(program.trainable_variables)}")

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Program serialization:")
    print("-" * 70)

    save_path = "examples/deep_synalinks_analyzer.program.json"
    program.save(save_path)
    print(f"Program saved to: {save_path}")

    loaded = synalinks.Program.load(save_path)
    print(f"Loaded successfully: {loaded.name}")

    # -------------------------------------------------------------------------
    # Key takeaways
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
This example demonstrates the FULL power of RLM integration:

1. MULTI-STAGE ITERATION: RLM iterates 15+ times, building
   understanding progressively (structure -> semantics -> relationships)

2. HYBRID ANALYSIS: Combines Python code execution (precise parsing)
   with llm_query() (semantic understanding) in the same loop

3. DUAL-MODEL ARCHITECTURE: Uses code_lm for generation and query_lm
   for semantic analysis (can use different model sizes)

4. RICH OUTPUT SCHEMA: DeepAnalysis captures structural, semantic,
   and relational insights; ComprehensiveSynopsis synthesizes them

5. INTEGRATED PROGRAM: RLM is one component in a Synalinks
   Program DAG, demonstrating composability

6. TRAINABLE: Both RLM instructions and Generator instructions
   can be optimized through Synalinks' training workflow

This is neuro-symbolic programming at its best: code execution for
precision, LLM reasoning for understanding, unified in a trainable
framework.
""")


async def cleanup_litellm():
    """Clean up litellm's background workers."""
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
        if GLOBAL_LOGGING_WORKER is not None:
            # Drain queued tasks explicitly to avoid un-awaited coroutine warnings.
            await GLOBAL_LOGGING_WORKER.clear_queue()
            await GLOBAL_LOGGING_WORKER.flush()
            running_tasks = list(getattr(GLOBAL_LOGGING_WORKER, "_running_tasks", []))
            if running_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*running_tasks, return_exceptions=True),
                        timeout=5,
                    )
                except asyncio.TimeoutError:
                    pass
            await GLOBAL_LOGGING_WORKER.stop()
    except Exception:
        pass


async def run():
    """Run with cleanup."""
    try:
        await main()
    finally:
        await cleanup_litellm()


if __name__ == "__main__":
    asyncio.run(run())
