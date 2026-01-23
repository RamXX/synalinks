#!/usr/bin/env python
# License Apache 2.0: (c) 2025 Synalinks Team

"""Example usage of RLM for iterative code-based reasoning.

This example demonstrates how to use RLM (Recursive Language Model) to
analyze documents by having the LLM write Python code iteratively.

Requirements:
    - Set GROQ_API_KEY environment variable
    - pip install synalinks

Usage:
    uv run --env-file .env -- python examples/repl_module_example.py
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
# Data Models
# =============================================================================


class DocumentQuery(synalinks.DataModel):
    """Input for document analysis."""

    document: str = synalinks.Field(description="The document to analyze")
    question: str = synalinks.Field(description="Question to answer about the document")


class AnalysisResult(synalinks.DataModel):
    """Output from document analysis."""

    answer: str = synalinks.Field(description="Answer to the question")
    evidence: list = synalinks.Field(description="Supporting evidence from the document")
    confidence: float = synalinks.Field(description="Confidence score 0-1")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run the RLM example."""
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        return

    # Create language model
    language_model = create_groq_language_model(
        "moonshotai/kimi-k2-instruct-0905",
        timeout=120,
    )

    # Create RLM module
    rlm = synalinks.modules.RLM(
        data_model=AnalysisResult,
        language_model=language_model,
        max_iterations=10,
        max_llm_calls=20,
        return_history=True,
        instructions="""
You are analyzing a document to answer a question.

Available variables:
- document: The full document text
- question: The question to answer

Use Python code to:
1. Explore the document structure
2. Search for relevant sections
3. Use llm_query() for semantic analysis of specific parts
4. Submit your final answer with SUBMIT(answer='...', evidence=[...], confidence=0.X)

Always print intermediate results to track your progress.
""",
    )

    # Build program using functional API
    inputs = synalinks.Input(data_model=DocumentQuery)
    outputs = await rlm(inputs)
    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="document_analyzer",
        description="Analyzes documents using iterative code execution",
    )

    # Example document
    sample_document = """
# Annual Report 2024

## Executive Summary
This year marked significant growth for our company. Revenue increased by 23%
compared to the previous year, reaching $45.2 million. Our customer base
expanded to over 50,000 active users.

## Key Achievements
- Launched 3 new product lines
- Expanded to 5 new international markets
- Reduced operational costs by 15%
- Achieved 98% customer satisfaction rating

## Financial Highlights
Total Revenue: $45.2 million (up 23% YoY)
Operating Margin: 18.5%
Net Income: $8.4 million
R&D Investment: $6.2 million (14% of revenue)

## Looking Forward
We expect continued growth in 2025, with projected revenue of $55 million
and plans to launch our enterprise platform in Q2.
"""

    # Run the analysis
    result = await program(
        DocumentQuery(
            document=sample_document,
            question="What was the company's revenue and how much did it grow?",
        )
    )

    # Display results
    print("=" * 60)
    print("ANALYSIS RESULT")
    print("=" * 60)
    print(f"\nAnswer: {result.get('answer')}")
    print(f"\nEvidence: {result.get('evidence')}")
    print(f"\nConfidence: {result.get('confidence')}")

    # Show execution history if available
    if "_history" in result.get_json():
        print("\n" + "=" * 60)
        print("EXECUTION HISTORY")
        print("=" * 60)
        for entry in result.get("_history"):
            print(f"\n--- Iteration {entry['iteration']} ---")
            reasoning = entry['reasoning'][:100] + "..." if len(entry['reasoning']) > 100 else entry['reasoning']
            print(f"Reasoning: {reasoning}")
            print(f"Code:\n{entry['code']}")
            if entry["stdout"]:
                stdout = entry['stdout'][:200] + "..." if len(entry['stdout']) > 200 else entry['stdout']
                print(f"Output: {stdout}")
            if entry["error"]:
                print(f"Error: {entry['error']}")


if __name__ == "__main__":
    asyncio.run(main())
