"""
Arithmetic Accuracy Test: ChainOfThought vs RecursiveChainOfThought

This is a focused test demonstrating RLM's key advantage: calculation accuracy
through code execution.

LLMs are notoriously bad at arithmetic, especially multi-step calculations.
RecursiveChainOfThought can execute Python code to verify calculations,
dramatically improving accuracy.

This test runs multiple arithmetic problems and compares accuracy rates.

Usage:
    export GROQ_API_KEY=your_key
    python examples/rlm_comparison/arithmetic_accuracy_test.py
"""

import asyncio
import os
import time
from dataclasses import dataclass

import synalinks


class MathProblem(synalinks.DataModel):
    """Input for math problems."""

    problem: str = synalinks.Field(description="The math problem to solve")


class MathResult(synalinks.DataModel):
    """Output for math problems."""

    answer: float = synalinks.Field(description="The numerical answer")


# Test problems with known answers
PROBLEMS = [
    # Basic but multi-step
    ("Calculate: 17 * 23 + 45 * 12 - 89", 17 * 23 + 45 * 12 - 89),  # 842
    ("What is 156 * 34 / 12 + 78?", 156 * 34 / 12 + 78),  # 520
    ("Compute: (125 + 375) * 4 - 1200", (125 + 375) * 4 - 1200),  # 800
    # Compound interest style
    (
        "If you invest $1000 at 5% annual interest compounded yearly, "
        "what is the value after 3 years? Round to 2 decimal places.",
        round(1000 * (1.05**3), 2),
    ),  # 1157.63
    # Sequential operations
    (
        "Start with 100. Multiply by 3. Subtract 50. Divide by 5. Add 30. "
        "What is the final result?",
        (100 * 3 - 50) / 5 + 30,
    ),  # 80
    # Percentages
    (
        "A store has 240 items. 35% are electronics, 25% are clothing, "
        "and the rest are home goods. How many home goods are there?",
        240 * 0.40,
    ),  # 96
    # Multi-step word problem
    (
        "A factory produces 450 units per day. Each unit costs $23 to make. "
        "If they operate 22 days per month, what is the monthly production cost?",
        450 * 23 * 22,
    ),  # 227,700
    # Fractions
    (
        "A recipe calls for 2/3 cup of sugar. If you're making 2.5 times "
        "the recipe, how many cups of sugar do you need? Give decimal answer.",
        (2 / 3) * 2.5,
    ),  # 1.667
]


@dataclass
class TestResult:
    """Result of a single test."""

    problem: str
    expected: float
    cot_answer: float | None
    cot_correct: bool
    cot_error: str | None
    rcot_answer: float | None
    rcot_correct: bool
    rcot_error: str | None


def is_close(a: float | None, b: float, tolerance: float = 0.01) -> bool:
    """Check if two numbers are close enough (within tolerance)."""
    if a is None:
        return False
    return abs(a - b) <= abs(b * tolerance) + 0.01


async def test_chain_of_thought(
    problem: str,
    language_model,
) -> tuple[float | None, str | None]:
    """Test ChainOfThought on a single problem."""
    try:
        inputs = synalinks.Input(data_model=MathProblem)
        outputs = await synalinks.ChainOfThought(
            data_model=MathResult,
            language_model=language_model,
            k=2,
            instructions=(
                "Solve this math problem step by step. "
                "Show your work in the thinking field, then give the numerical answer."
            ),
        )(inputs)

        program = synalinks.Program(inputs=inputs, outputs=outputs)

        query = MathProblem(problem=problem)
        result = await program(query.to_json_data_model())

        if result and result.json:
            import json

            data = json.loads(result.json)
            return float(data.get("answer", 0)), None

        return None, "No result"

    except Exception as e:
        return None, str(e)


async def test_recursive_chain_of_thought(
    problem: str,
    language_model,
) -> tuple[float | None, str | None]:
    """Test RecursiveChainOfThought on a single problem."""
    try:
        inputs = synalinks.Input(data_model=MathProblem)

        rcot = synalinks.RecursiveChainOfThought(
            data_model=MathResult,
            language_model=language_model,
            k=2,
            max_iterations=10,
            instructions=(
                "Solve this math problem. "
                "IMPORTANT: Use Python code in a ```repl block to calculate the answer. "
                "Do NOT do mental math - execute the calculation in code to ensure accuracy. "
                "When you have the answer, use FINAL_VAR(result) where result contains "
                'a dict like {"answer": <number>}.'
            ),
        )

        outputs = await rcot(inputs)
        program = synalinks.Program(inputs=inputs, outputs=outputs)

        query = MathProblem(problem=problem)
        result = await program(query.to_json_data_model())

        if result and result.json:
            import json

            data = json.loads(result.json)
            return float(data.get("answer", 0)), None

        return None, "No result"

    except Exception as e:
        return None, str(e)


async def run_comparison(language_model, num_problems: int = None):
    """Run the full comparison test."""
    problems_to_test = PROBLEMS[:num_problems] if num_problems else PROBLEMS
    results: list[TestResult] = []

    print(f"\nRunning {len(problems_to_test)} arithmetic tests...")
    print("-" * 70)

    for i, (problem, expected) in enumerate(problems_to_test, 1):
        print(f"\n[{i}/{len(problems_to_test)}] {problem[:50]}...")

        # Test ChainOfThought
        print("  ChainOfThought: ", end="", flush=True)
        cot_answer, cot_error = await test_chain_of_thought(problem, language_model)
        cot_correct = is_close(cot_answer, expected)
        if cot_error:
            print(f"ERROR - {cot_error[:30]}")
        else:
            status = "CORRECT" if cot_correct else "WRONG"
            print(f"{cot_answer} (expected {expected}) - {status}")

        # Test RecursiveChainOfThought
        print("  RecursiveCoT:   ", end="", flush=True)
        rcot_answer, rcot_error = await test_recursive_chain_of_thought(
            problem, language_model
        )
        rcot_correct = is_close(rcot_answer, expected)
        if rcot_error:
            print(f"ERROR - {rcot_error[:30]}")
        else:
            status = "CORRECT" if rcot_correct else "WRONG"
            print(f"{rcot_answer} (expected {expected}) - {status}")

        results.append(
            TestResult(
                problem=problem,
                expected=expected,
                cot_answer=cot_answer,
                cot_correct=cot_correct,
                cot_error=cot_error,
                rcot_answer=rcot_answer,
                rcot_correct=rcot_correct,
                rcot_error=rcot_error,
            )
        )

    return results


def print_summary(results: list[TestResult]):
    """Print summary statistics."""
    cot_correct = sum(1 for r in results if r.cot_correct)
    cot_errors = sum(1 for r in results if r.cot_error)
    rcot_correct = sum(1 for r in results if r.rcot_correct)
    rcot_errors = sum(1 for r in results if r.rcot_error)
    total = len(results)

    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON RESULTS")
    print("=" * 70)

    print(f"\nChainOfThought:")
    print(f"  Correct: {cot_correct}/{total} ({100*cot_correct/total:.1f}%)")
    print(f"  Errors:  {cot_errors}/{total}")

    print(f"\nRecursiveChainOfThought:")
    print(f"  Correct: {rcot_correct}/{total} ({100*rcot_correct/total:.1f}%)")
    print(f"  Errors:  {rcot_errors}/{total}")

    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    if rcot_correct > cot_correct:
        improvement = rcot_correct - cot_correct
        print(
            f"\nRecursiveChainOfThought was MORE ACCURATE by {improvement} problems"
        )
        print(
            "This demonstrates the advantage of code execution for arithmetic tasks."
        )
    elif rcot_correct == cot_correct:
        print("\nBoth approaches had the same accuracy.")
    else:
        print("\nChainOfThought performed better on this run.")

    # Show which problems each got wrong
    print("\n" + "-" * 70)
    print("DETAILED COMPARISON")
    print("-" * 70)

    for r in results:
        cot_status = "PASS" if r.cot_correct else ("ERR" if r.cot_error else "FAIL")
        rcot_status = "PASS" if r.rcot_correct else ("ERR" if r.rcot_error else "FAIL")
        print(f"\n{r.problem[:60]}...")
        print(f"  Expected: {r.expected}")
        print(f"  CoT: {r.cot_answer} [{cot_status}]")
        print(f"  RCoT: {r.rcot_answer} [{rcot_status}]")


async def main():
    """Main test runner."""
    print("=" * 70)
    print("ARITHMETIC ACCURACY TEST")
    print("ChainOfThought vs RecursiveChainOfThought")
    print("=" * 70)

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("\nError: GROQ_API_KEY environment variable required")
        print("Export your Groq API key and try again.")
        return

    # Use groq model for both (fast and cheap for testing)
    language_model = synalinks.LanguageModel(model="groq/openai/gpt-oss-20b")
    print(f"\nUsing model: groq/openai/gpt-oss-20b")

    # Run comparison
    results = await run_comparison(language_model, num_problems=5)  # Quick test

    # Print summary
    print_summary(results)

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print(
        """
RecursiveChainOfThought uses Python code execution to verify calculations,
while regular ChainOfThought relies entirely on the LLM's internal reasoning.

LLMs are notoriously unreliable at arithmetic, especially:
- Multi-step calculations
- Decimal/fraction operations
- Order of operations

By executing actual Python code, RecursiveChainOfThought eliminates these
errors and provides verified, accurate results.
"""
    )


if __name__ == "__main__":
    synalinks.clear_session()
    asyncio.run(main())
