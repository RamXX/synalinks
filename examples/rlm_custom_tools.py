#!/usr/bin/env python
# License Apache 2.0: (c) 2025 Synalinks Team

"""RLM Example: Custom Tools

This example demonstrates using RLM (Recursive Language Model) with custom tools
that extend the LLM's capabilities beyond basic Python execution.

Custom tools extend the RLM's capabilities beyond basic Python execution,
allowing it to query databases, call APIs, or perform domain-specific operations.

Requirements:
    - Set GROQ_API_KEY environment variable
    - pip install synalinks

Usage:
    uv run --env-file .env -- python examples/rlm_custom_tools.py
"""

import asyncio
import os

import synalinks
from synalinks.src.modules.reasoning.repl_module import RLM


# =============================================================================
# Custom Tools - These extend what the LLM can do in the REPL
# =============================================================================

# Register tools for serialization
from synalinks.saving import register_synalinks_serializable

# Simulated database for the example
MOCK_DATABASE = {
    "users": [
        {"id": 1, "name": "Alice Chen", "email": "alice@example.com", "plan": "premium", "signup_date": "2024-01-15", "monthly_spend": 299.99},
        {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "plan": "basic", "signup_date": "2024-02-20", "monthly_spend": 49.99},
        {"id": 3, "name": "Carol White", "email": "carol@example.com", "plan": "premium", "signup_date": "2024-03-10", "monthly_spend": 299.99},
        {"id": 4, "name": "David Brown", "email": "david@example.com", "plan": "enterprise", "signup_date": "2024-01-05", "monthly_spend": 999.99},
        {"id": 5, "name": "Eve Johnson", "email": "eve@example.com", "plan": "basic", "signup_date": "2024-04-01", "monthly_spend": 49.99},
        {"id": 6, "name": "Frank Lee", "email": "frank@example.com", "plan": "premium", "signup_date": "2024-02-28", "monthly_spend": 299.99},
    ],
    "transactions": [
        {"user_id": 1, "amount": 299.99, "date": "2024-04-01", "type": "subscription"},
        {"user_id": 1, "amount": 50.00, "date": "2024-04-05", "type": "addon"},
        {"user_id": 2, "amount": 49.99, "date": "2024-04-01", "type": "subscription"},
        {"user_id": 3, "amount": 299.99, "date": "2024-04-01", "type": "subscription"},
        {"user_id": 4, "amount": 999.99, "date": "2024-04-01", "type": "subscription"},
        {"user_id": 4, "amount": 500.00, "date": "2024-04-10", "type": "addon"},
        {"user_id": 5, "amount": 49.99, "date": "2024-04-01", "type": "subscription"},
        {"user_id": 6, "amount": 299.99, "date": "2024-04-01", "type": "subscription"},
    ],
}


@register_synalinks_serializable()
async def search_users(query: str) -> dict:
    """Search users by name, email, or plan.

    Args:
        query (str): Search term to match against user fields.
    """
    query_lower = query.lower()
    results = []
    for user in MOCK_DATABASE["users"]:
        if (query_lower in user["name"].lower() or
            query_lower in user["email"].lower() or
            query_lower in user["plan"].lower()):
            results.append(user)
    return {"results": results, "count": len(results)}


@register_synalinks_serializable()
async def get_user_transactions(user_id: int) -> dict:
    """Get all transactions for a specific user.

    Args:
        user_id (int): The user's ID.
    """
    results = [t for t in MOCK_DATABASE["transactions"] if t["user_id"] == user_id]
    return {"transactions": results, "count": len(results)}


@register_synalinks_serializable()
async def calculate_mrr() -> dict:
    """Calculate Monthly Recurring Revenue breakdown by plan."""
    mrr_by_plan = {}
    for user in MOCK_DATABASE["users"]:
        plan = user["plan"]
        mrr_by_plan[plan] = mrr_by_plan.get(plan, 0) + user["monthly_spend"]

    total_mrr = sum(mrr_by_plan.values())
    return {
        "total_mrr": round(total_mrr, 2),
        "by_plan": {k: round(v, 2) for k, v in mrr_by_plan.items()},
        "user_count": len(MOCK_DATABASE["users"]),
    }


@register_synalinks_serializable()
async def get_churn_risk(threshold: float = 30.0) -> dict:
    """Identify users at risk of churning based on low spend.

    Args:
        threshold (float): Monthly spend threshold below which users are at risk.
    """
    at_risk = [u for u in MOCK_DATABASE["users"] if u["monthly_spend"] < threshold]
    return {"at_risk_users": at_risk, "count": len(at_risk)}


# =============================================================================
# Example
# =============================================================================


class BusinessQuery(synalinks.DataModel):
    """Input for business intelligence queries."""
    question: str = synalinks.Field(description="Business question to answer")


class BusinessInsight(synalinks.DataModel):
    """Output with business insights."""
    answer: str = synalinks.Field(description="Answer to the business question")
    data_sources: str = synalinks.Field(description="Which tools/data sources were used")
    recommendation: str = synalinks.Field(description="Actionable recommendation based on findings")


async def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        return

    # Create language model (Groq support is built into LanguageModel)
    lm = synalinks.LanguageModel(model="groq/moonshotai/kimi-k2-instruct-0905", timeout=120)

    # Define custom tools as Tool objects (Synalinks pattern)
    custom_tools = [
        synalinks.Tool(search_users),
        synalinks.Tool(get_user_transactions),
        synalinks.Tool(calculate_mrr),
        synalinks.Tool(get_churn_risk),
    ]

    # Create RLM with custom tools
    rlm = RLM(
        data_model=BusinessInsight,
        language_model=lm,
        max_iterations=10,
        tools=custom_tools,
        return_history=True,
        instructions="""
You are a business intelligence analyst with access to company data.

Available tools (call as functions in your Python code):
- search_users(query) - Search users by name, email, or plan type
- get_user_transactions(user_id) - Get transactions for a specific user
- calculate_mrr() - Calculate Monthly Recurring Revenue by plan
- get_churn_risk(threshold=30.0) - Find users with spend below threshold

Use these tools to answer business questions. Always:
1. Explore the data first using the tools
2. Print results to understand the data
3. Perform any needed calculations
4. Submit with SUBMIT(answer='...', data_sources='...', recommendation='...')
""",
    )

    inputs = synalinks.Input(data_model=BusinessQuery)
    outputs = await rlm(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="business_analyst")

    print("=" * 70)
    print("CUSTOM TOOLS EXAMPLE - Business Intelligence")
    print("=" * 70)
    print("\nAvailable tools:")
    for tool in custom_tools:
        print(f"  - {tool.name}: {tool.description}")

    # Run business queries
    questions = [
        "What is our current MRR and how is it distributed across plans?",
        "Who are our premium users and what is their total lifetime value based on transactions?",
        "Which users are at risk of churning and what should we do about it?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"QUERY {i}: {question}")
        print("=" * 70)

        result = await program(BusinessQuery(question=question))

        if result:
            print(f"\nAnswer: {result.get('answer')}")
            print(f"\nData Sources: {result.get('data_sources')}")
            print(f"\nRecommendation: {result.get('recommendation')}")

            json_data = result.get_json()
            if "_history" in json_data:
                print(f"\n[Completed in {len(json_data['_history'])} iterations]")
        else:
            print("Error: No result returned")


if __name__ == "__main__":
    asyncio.run(main())
