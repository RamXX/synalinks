#!/usr/bin/env python
# License Apache 2.0: (c) 2025 Synalinks Team

"""RLM Example: Data Analysis with Multi-step Computation

This example demonstrates using RLM (Recursive Language Model) for statistical
analysis over structured data, requiring multiple computational steps.

Equivalent to DSPy's RLM with complex data:
    rlm = dspy.RLM("data, question -> result: float, explanation: str")
    result = rlm(data=sales_data, question="What is the average growth rate?")

Requirements:
    - Set GROQ_API_KEY environment variable
    - pip install synalinks

Usage:
    uv run --env-file .env -- python examples/rlm_data_analysis.py
"""

import asyncio
import copy
import json
import os
import warnings

import litellm

import synalinks
from synalinks.src.backend import ChatRole
from synalinks.src.language_models.language_model import LanguageModel
from synalinks.src.modules.reasoning.repl_module import RLM
from synalinks.src.utils.nlp_utils import shorten_text


# =============================================================================
# Groq Workaround
# =============================================================================


def _clean_messages_for_groq(messages: list) -> list:
    cleaned = []
    for msg in messages:
        clean_msg = {"role": msg.get("role"), "content": msg.get("content", "")}
        cleaned.append(clean_msg)
    return cleaned


_original_call = None


async def _patched_call(self, messages, schema=None, streaming=False, **kwargs):
    formatted_messages = messages.get_json().get("messages", [])
    input_kwargs = copy.deepcopy(kwargs)
    schema = copy.deepcopy(schema)

    if self.model.startswith("groq"):
        formatted_messages = _clean_messages_for_groq(formatted_messages)

    if schema and self.model.startswith("groq"):
        kwargs.update({
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "schema": schema},
            }
        })

    for i in range(self.retry):
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=formatted_messages,
                timeout=self.timeout,
                caching=self.caching,
                **kwargs,
            )
            response_str = response["choices"][0]["message"]["content"].strip()
            return json.loads(response_str) if schema else {"role": ChatRole.ASSISTANT, "content": response_str}
        except Exception as e:
            warnings.warn(f"Error: {shorten_text(str(e))}")
            await asyncio.sleep(1)
    return None


def patch_synalinks_for_groq():
    global _original_call
    if _original_call is None:
        _original_call = LanguageModel.__call__
        LanguageModel.__call__ = _patched_call


def create_groq_language_model(model_name: str, **kwargs) -> synalinks.LanguageModel:
    patch_synalinks_for_groq()
    return synalinks.LanguageModel(model=f"groq/{model_name}", **kwargs)


# =============================================================================
# Example: Data Analysis
# =============================================================================


class DataAnalysisInput(synalinks.DataModel):
    """Input for data analysis."""
    data: dict = synalinks.Field(description="The structured data to analyze")
    question: str = synalinks.Field(description="Analysis question to answer")


class AnalysisOutput(synalinks.DataModel):
    """Output with numeric result and explanation."""
    result: float = synalinks.Field(description="Numeric result of the analysis")
    methodology: str = synalinks.Field(description="How the result was computed")
    insights: str = synalinks.Field(description="Key insights from the analysis")


async def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        return

    lm = create_groq_language_model("moonshotai/kimi-k2-instruct-0905", timeout=120)

    # Create RLM for data analysis
    rlm = RLM(
        data_model=AnalysisOutput,
        language_model=lm,
        max_iterations=15,
        return_history=True,
        instructions="""
You are a data analyst. Analyze the provided data to answer the question.

Available variables:
- data: A dictionary containing the structured data
- question: The analysis question to answer

Steps:
1. First explore the data structure (print keys, sample values)
2. Identify relevant fields for the analysis
3. Perform calculations step by step
4. Verify your results make sense
5. Submit with SUBMIT(result=number, methodology='...', insights='...')

Use Python for all calculations. Always print intermediate results.
""",
    )

    inputs = synalinks.Input(data_model=DataAnalysisInput)
    outputs = await rlm(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="data_analyzer")

    # Example: Sales data analysis
    sales_data = {
        "quarters": ["Q1", "Q2", "Q3", "Q4"],
        "revenue": [125000, 148000, 162000, 195000],
        "costs": [95000, 108000, 115000, 132000],
        "units_sold": [1250, 1480, 1620, 1950],
        "regions": {
            "north": {"revenue": 180000, "customers": 450},
            "south": {"revenue": 220000, "customers": 380},
            "east": {"revenue": 150000, "customers": 290},
            "west": {"revenue": 80000, "customers": 180},
        },
        "products": [
            {"name": "Widget A", "price": 99.99, "units": 2500, "margin": 0.35},
            {"name": "Widget B", "price": 149.99, "units": 1800, "margin": 0.42},
            {"name": "Widget C", "price": 249.99, "units": 900, "margin": 0.55},
            {"name": "Widget D", "price": 49.99, "units": 3100, "margin": 0.25},
        ],
    }

    print("=" * 70)
    print("DATA ANALYSIS EXAMPLE")
    print("=" * 70)
    print("\nData structure:")
    print(f"  - Quarterly data: {len(sales_data['quarters'])} quarters")
    print(f"  - Regions: {len(sales_data['regions'])} regions")
    print(f"  - Products: {len(sales_data['products'])} products")

    # Run multiple analyses
    questions = [
        "Calculate the total profit (revenue - costs) across all quarters and the profit margin percentage",
        "Which region has the highest revenue per customer?",
        "Calculate the weighted average profit margin across all products (weighted by units sold)",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"ANALYSIS {i}: {question}")
        print("=" * 70)

        result = await program(
            DataAnalysisInput(data=sales_data, question=question)
        )

        if result:
            print(f"\nResult: {result.get('result')}")
            print(f"Methodology: {result.get('methodology')}")
            print(f"Insights: {result.get('insights')}")

            # Show trajectory summary
            json_data = result.get_json()
            if "_history" in json_data:
                print(f"\n[Completed in {len(json_data['_history'])} iterations]")
        else:
            print("Error: No result returned")


if __name__ == "__main__":
    asyncio.run(main())
