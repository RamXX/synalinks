#!/usr/bin/env python
# License Apache 2.0: (c) 2025 Synalinks Team

"""RLM Example: Structured Data Extraction

This example demonstrates using RLM (Recursive Language Model) to extract
structured information from unstructured text, producing multiple typed output fields.

Equivalent to DSPy's RLM with multiple outputs:
    rlm = dspy.RLM("document -> company: str, revenue: float, employees: int, founded: int")
    result = rlm(document="...")
    print(result.company, result.revenue)

Requirements:
    - Set GROQ_API_KEY environment variable
    - pip install synalinks

Usage:
    uv run --env-file .env -- python examples/rlm_structured_extraction.py
"""

import asyncio
import copy
import json
import os
import warnings
from typing import List

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
# Example: Structured Extraction
# =============================================================================


class DocumentInput(synalinks.DataModel):
    """Input document for extraction."""
    document: str = synalinks.Field(description="The document to extract information from")


class CompanyProfile(synalinks.DataModel):
    """Structured company profile extracted from text."""
    company_name: str = synalinks.Field(description="Name of the company")
    industry: str = synalinks.Field(description="Industry sector")
    founded_year: int = synalinks.Field(description="Year the company was founded")
    headquarters: str = synalinks.Field(description="City and country of headquarters")
    employee_count: int = synalinks.Field(description="Number of employees")
    annual_revenue: float = synalinks.Field(description="Annual revenue in millions USD")
    key_products: list = synalinks.Field(description="List of main products or services")
    competitors: list = synalinks.Field(description="List of main competitors")


class JobPosting(synalinks.DataModel):
    """Structured job posting extracted from text."""
    job_title: str = synalinks.Field(description="Title of the position")
    company: str = synalinks.Field(description="Company name")
    location: str = synalinks.Field(description="Job location")
    salary_min: float = synalinks.Field(description="Minimum salary in thousands USD")
    salary_max: float = synalinks.Field(description="Maximum salary in thousands USD")
    experience_years: int = synalinks.Field(description="Required years of experience")
    skills: list = synalinks.Field(description="Required skills")
    benefits: list = synalinks.Field(description="Listed benefits")
    remote_friendly: bool = synalinks.Field(description="Whether remote work is allowed")


async def run_company_extraction(lm):
    """Extract company profile from unstructured text."""
    rlm = RLM(
        data_model=CompanyProfile,
        language_model=lm,
        max_iterations=10,
        return_history=True,
        instructions="""
Extract structured company information from the document.

Available variables:
- document: The text to extract from

Steps:
1. Read through the document carefully
2. Use string operations to find specific data points
3. Use llm_query() if you need to interpret ambiguous text
4. Extract all required fields
5. Submit with SUBMIT(company_name='...', industry='...', ...)

Be precise with numbers. Convert text like "1.2 billion" to numeric values.
""",
    )

    inputs = synalinks.Input(data_model=DocumentInput)
    outputs = await rlm(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="company_extractor")

    document = """
    Acme Technologies Inc.

    About Us:
    Founded in 2015 by Sarah Chen and Michael Park, Acme Technologies has grown from
    a small startup in Austin, Texas to a global leader in cloud infrastructure.
    Our headquarters remains in Austin, with offices in London, Singapore, and Tokyo.

    Company Overview:
    - Industry: Cloud Computing & Infrastructure
    - Employees: Approximately 2,400 team members worldwide
    - Annual Revenue: $340 million (FY 2024)
    - Funding: Series D ($180M raised to date)

    Our Products:
    We offer three main product lines:
    1. CloudSync Pro - Enterprise file synchronization
    2. InfraManager - Cloud infrastructure management
    3. SecureVault - Zero-trust security platform
    4. DataPipeline - ETL and data integration tools

    Market Position:
    We compete primarily with Amazon Web Services (AWS), Microsoft Azure,
    Google Cloud Platform (GCP), and emerging players like HashiCorp and Datadog.
    Our focus on developer experience sets us apart.

    Contact: info@acmetech.example.com
    """

    print("=" * 70)
    print("EXTRACTION 1: Company Profile")
    print("=" * 70)
    print(f"Document length: {len(document)} characters")

    result = await program(DocumentInput(document=document))

    if result:
        print("\nExtracted Profile:")
        print(f"  Company: {result.get('company_name')}")
        print(f"  Industry: {result.get('industry')}")
        print(f"  Founded: {result.get('founded_year')}")
        print(f"  HQ: {result.get('headquarters')}")
        emp = result.get('employee_count')
        print(f"  Employees: {emp:,}" if emp else "  Employees: N/A")
        rev = result.get('annual_revenue')
        print(f"  Revenue: ${rev}M" if rev else "  Revenue: N/A")
        print(f"  Products: {result.get('key_products')}")
        print(f"  Competitors: {result.get('competitors')}")

        json_data = result.get_json()
        if "_history" in json_data:
            print(f"\n[Completed in {len(json_data['_history'])} iterations]")


async def run_job_extraction(lm):
    """Extract job posting details from unstructured text."""
    rlm = RLM(
        data_model=JobPosting,
        language_model=lm,
        max_iterations=10,
        return_history=True,
        instructions="""
Extract structured job posting information from the document.

Available variables:
- document: The job posting text

Parse salary ranges carefully (e.g., "$120K-$150K" means min=120, max=150).
Look for remote work mentions like "remote", "hybrid", "work from home".
Extract skills from requirements sections.
Submit with SUBMIT(job_title='...', company='...', ..., remote_friendly=True/False).
""",
    )

    inputs = synalinks.Input(data_model=DocumentInput)
    outputs = await rlm(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="job_extractor")

    job_posting = """
    Senior Machine Learning Engineer - AI Platform Team

    Company: DataMind AI
    Location: San Francisco, CA (Hybrid - 2 days in office)

    About the Role:
    We're looking for a Senior ML Engineer to join our AI Platform team.
    You'll work on building and scaling our core ML infrastructure that
    powers recommendations for 50M+ users.

    Compensation:
    - Base Salary: $180,000 - $240,000 depending on experience
    - Equity: 0.05% - 0.15%
    - Annual Bonus: 15-20% of base

    Requirements:
    - 5+ years of experience in machine learning engineering
    - Strong Python skills (PyTorch, TensorFlow, or JAX)
    - Experience with distributed systems (Kubernetes, Ray, Spark)
    - Background in recommender systems or NLP preferred
    - MS/PhD in Computer Science, ML, or related field

    Nice to Have:
    - Experience with LLMs and foundation models
    - Publications in top ML venues
    - Open source contributions

    Benefits:
    - Comprehensive health, dental, and vision insurance
    - Unlimited PTO policy
    - $5,000 annual learning budget
    - 401(k) with 4% match
    - Gym membership reimbursement
    - Catered lunches (in-office days)
    - Parental leave (16 weeks)

    Remote Policy:
    This role is hybrid with flexibility. We require 2 days per week in our
    SF office for team collaboration, but you can work remotely the other days.
    Occasional remote work weeks are fine.

    Apply at: careers.datamind.ai/ml-engineer
    """

    print("\n" + "=" * 70)
    print("EXTRACTION 2: Job Posting")
    print("=" * 70)
    print(f"Document length: {len(job_posting)} characters")

    result = await program(DocumentInput(document=job_posting))

    if result:
        print("\nExtracted Job Details:")
        print(f"  Title: {result.get('job_title')}")
        print(f"  Company: {result.get('company')}")
        print(f"  Location: {result.get('location')}")
        sal_min = result.get('salary_min')
        sal_max = result.get('salary_max')
        if sal_min and sal_max:
            print(f"  Salary: ${sal_min}K - ${sal_max}K")
        else:
            print(f"  Salary: {sal_min} - {sal_max}")
        print(f"  Experience: {result.get('experience_years')}+ years")
        print(f"  Skills: {result.get('skills')}")
        print(f"  Benefits: {result.get('benefits')}")
        print(f"  Remote Friendly: {result.get('remote_friendly')}")

        json_data = result.get_json()
        if "_history" in json_data:
            print(f"\n[Completed in {len(json_data['_history'])} iterations]")


async def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        return

    lm = create_groq_language_model("moonshotai/kimi-k2-instruct-0905", timeout=120)

    print("=" * 70)
    print("STRUCTURED EXTRACTION EXAMPLE")
    print("=" * 70)
    print("\nThis example demonstrates extracting typed, structured data")
    print("from unstructured documents using iterative code execution.\n")

    await run_company_extraction(lm)
    await run_job_extraction(lm)


if __name__ == "__main__":
    asyncio.run(main())
