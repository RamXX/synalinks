#!/usr/bin/env python
# License Apache 2.0: (c) 2025 Synalinks Team

"""RLM Example: Document Analysis

This example demonstrates using RLM (Recursive Language Model) to analyze a long
document by having the LLM write Python code to explore and extract information.

The RLM writes and executes Python code iteratively to explore and
extract information from the document before submitting a final answer.

Requirements:
    - Set GROQ_API_KEY environment variable
    - pip install synalinks

Usage:
    uv run --env-file .env -- python examples/rlm_document_analysis.py
"""

import asyncio
import os

import synalinks
from synalinks.src.modules.reasoning.repl_module import RLM


# =============================================================================
# Example: Document Analysis
# =============================================================================


# Define input/output data models
class DocumentQuery(synalinks.DataModel):
    """Input: document and question to answer about it."""
    context: str = synalinks.Field(description="The document to analyze")
    query: str = synalinks.Field(description="Question to answer about the document")


class AnalysisResult(synalinks.DataModel):
    """Output: answer with supporting evidence."""
    answer: str = synalinks.Field(description="Answer to the question")
    evidence: str = synalinks.Field(description="Supporting evidence from the document")


async def main():
    """Run the document analysis example."""
    # Check for API key
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        return

    # Create language model (Groq support is built into LanguageModel)
    lm = synalinks.LanguageModel(model="groq/moonshotai/kimi-k2-instruct-0905", timeout=120)

    # Create RLM for document analysis
    rlm = RLM(
        data_model=AnalysisResult,
        language_model=lm,
        max_iterations=10,
        max_llm_calls=20,
        return_history=True,
        instructions="""
You are analyzing a document to answer a question.

Available variables:
- context: The full document text
- query: The question to answer

Use Python code to:
1. Explore the document structure (print sections, count paragraphs, etc.)
2. Search for relevant sections using string operations
3. Use llm_query(prompt) for semantic analysis of specific parts
4. Submit your final answer with SUBMIT(answer='...', evidence='...')

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

    # Example: Analyze a company annual report
    sample_document = """
# TechCorp Annual Report 2024

## Executive Summary
This year marked significant growth for TechCorp. Revenue increased by 23%
compared to the previous year, reaching $45.2 million. Our customer base
expanded to over 50,000 active users across 12 countries.

## Key Achievements
- Launched 3 new product lines: CloudSync, DataVault, and AIAssist
- Expanded to 5 new international markets: Germany, Japan, Brazil, India, Australia
- Reduced operational costs by 15% through automation
- Achieved 98% customer satisfaction rating (up from 94% last year)
- Hired 150 new employees, growing team to 500 total

## Financial Highlights
Total Revenue: $45.2 million (up 23% YoY)
Operating Margin: 18.5%
Net Income: $8.4 million
R&D Investment: $6.2 million (14% of revenue)
Marketing Spend: $4.1 million

## Product Performance
CloudSync: $18.5 million revenue (41% of total)
DataVault: $15.2 million revenue (34% of total)
AIAssist: $8.3 million revenue (18% of total)
Legacy Products: $3.2 million revenue (7% of total)

## Customer Metrics
Total Active Users: 50,000+
Enterprise Customers: 250
SMB Customers: 2,500
Churn Rate: 3.2% (improved from 4.8%)
Net Promoter Score: 72

## Looking Forward
We expect continued growth in 2025, with projected revenue of $55 million
and plans to launch our enterprise platform in Q2. Key initiatives include:
- AI-powered analytics dashboard
- Enhanced security features
- Mobile app redesign
- Partnership with major cloud providers
"""

    print("=" * 70)
    print("DOCUMENT ANALYSIS EXAMPLE")
    print("=" * 70)
    print(f"\nDocument length: {len(sample_document)} characters")
    print(f"Query: What is the company's best-performing product and how much revenue did it generate?")
    print("\nRunning RLM...")
    print("-" * 70)

    # Run the analysis
    result = await program(
        DocumentQuery(
            context=sample_document,
            query="What is the company's best-performing product and how much revenue did it generate?",
        )
    )

    # Display results
    print("\n" + "=" * 70)
    print("ANALYSIS RESULT")
    print("=" * 70)

    if result is None:
        print("Error: No result returned")
        return

    print(f"\nAnswer: {result.get('answer')}")
    print(f"\nEvidence: {result.get('evidence')}")

    # Show execution history
    json_data = result.get_json()
    if "_history" in json_data:
        print("\n" + "=" * 70)
        print("EXECUTION TRAJECTORY")
        print("=" * 70)
        for entry in json_data["_history"]:
            print(f"\n--- Iteration {entry['iteration']} ---")
            print(f"Reasoning: {entry['reasoning'][:150]}..." if len(entry['reasoning']) > 150 else f"Reasoning: {entry['reasoning']}")
            print(f"Code:\n{entry['code']}")
            if entry["stdout"]:
                stdout = entry["stdout"][:300] + "..." if len(entry["stdout"]) > 300 else entry["stdout"]
                print(f"Output: {stdout}")
            if entry["error"]:
                print(f"Error: {entry['error']}")


if __name__ == "__main__":
    asyncio.run(main())
