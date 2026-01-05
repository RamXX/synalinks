"""Training example for RecursiveGenerator with multi-model setup.

This example demonstrates how to:
1. Create a RecursiveGenerator with multi-model architecture (root + sub models)
2. Add seed instructions for OMEGA-style prompt evolution
3. Compile a Program with RecursiveGenerator using an optimizer
4. Train the program to optimize prompts via in-context learning

The multi-model setup optimizes costs by using:
- Root model (zai/glm-4.7): Orchestration and code generation
- Sub model (groq/openai/gpt-oss-20b): Cheaper recursive sub-calls
"""

import asyncio

import synalinks


# Define data models
class Query(synalinks.DataModel):
    """User query with large context."""

    question: str = synalinks.Field(description="The user's question")
    documents: list = synalinks.Field(
        description="Large set of documents to analyze"
    )


class Answer(synalinks.DataModel):
    """Structured answer."""

    answer: str = synalinks.Field(description="The answer to the question")
    confidence: float = synalinks.Field(
        description="Confidence score between 0 and 1"
    )
    sources: list = synalinks.Field(
        description="List of document indices used"
    )


async def main():
    """Training example with multi-model RecursiveGenerator."""

    # Multi-model setup for cost optimization
    # Root model: expensive but capable (orchestration)
    lm_root = synalinks.LanguageModel(model="zai/glm-4.7")

    # Sub model: cheaper for recursive calls
    lm_sub = synalinks.LanguageModel(model="groq/openai/gpt-oss-20b")

    # Seed instructions for OMEGA-style prompt evolution
    # The optimizer will explore variations of these instructions
    seed_instructions = [
        """You are a recursive document analyzer. Use the REPL to:
1. Iterate through documents systematically
2. Make recursive llm_query() calls to analyze chunks
3. Track relevant findings in variables
4. Synthesize results into structured output""",
        """Analyze documents using divide-and-conquer:
1. Split documents into logical chunks using code
2. Use llm_query() for parallel sub-analysis
3. Aggregate results and extract key insights
4. Build final answer with confidence scoring""",
        """Process large document sets recursively:
1. Use code to filter relevant documents first
2. Make targeted llm_query() calls for deep analysis
3. Maintain state in REPL variables
4. Construct comprehensive answer with sources""",
    ]

    # Few-shot examples (optional)
    # These demonstrate the expected input-output format
    examples = [
        (
            {
                "question": "What are the main themes?",
                "documents": ["Doc 1 about AI...", "Doc 2 about ML..."],
            },
            {
                "answer": "The main themes are AI and machine learning.",
                "confidence": 0.95,
                "sources": [0, 1],
            },
        ),
    ]

    # Create RecursiveGenerator with training support
    x0 = synalinks.Input(data_model=Query)
    x1 = await synalinks.RecursiveGenerator(
        data_model=Answer,
        language_model=lm_root,  # Root model for orchestration
        sub_language_model=lm_sub,  # Cheaper model for recursive calls
        seed_instructions=seed_instructions,  # Enable OMEGA optimization
        examples=examples,  # Few-shot examples (trainable)
        max_iterations=30,
        max_depth=2,  # Allow nested recursion
        trainable=True,  # Enable training (default)
    )(x0)

    # Build program
    program = synalinks.Program(
        inputs=x0,
        outputs=x1,
        name="recursive_document_analyzer",
        description="Analyze large document sets using RLM with optimized prompts",
    )

    # Compile with optimizer and reward
    # The optimizer will evolve prompts to maximize the reward
    program.compile(
        optimizer=synalinks.optimizers.GRPO(
            # GRPO: Group Relative Policy Optimization
            # Optimizes prompts based on relative performance
            learning_rate=0.01,
            batch_size=4,
        ),
        reward=synalinks.rewards.CosineSimilarity(),  # Semantic similarity reward
        metrics=[
            synalinks.metrics.MeanMetricWrapper(
                synalinks.rewards.exact_match, name="exact_match"
            ),
        ],
    )

    # Training data
    training_data = [
        Query(
            question="What are the key findings about climate change?",
            documents=[
                "Document 1: Global temperatures rising...",
                "Document 2: Ice sheets melting faster...",
                "Document 3: Extreme weather events increasing...",
            ],
        ),
        Query(
            question="Summarize the economic impacts mentioned.",
            documents=[
                "Document A: GDP growth slowing...",
                "Document B: Inflation concerns...",
                "Document C: Job market shifts...",
            ],
        ),
        # Add more training examples...
    ]

    training_targets = [
        Answer(
            answer="Key findings: temperatures rising, ice melting, extreme weather",
            confidence=0.92,
            sources=[0, 1, 2],
        ),
        Answer(
            answer="Economic impacts: GDP slowdown, inflation, job market changes",
            confidence=0.88,
            sources=[0, 1, 2],
        ),
        # Corresponding targets...
    ]

    # Train the program
    # The optimizer will:
    # 1. Explore different prompt variations from seed_instructions
    # 2. Evaluate performance using CosineSimilarity reward
    # 3. Optimize to find best-performing prompts
    # 4. Update trainable variables (instructions, examples)
    history = await program.fit(
        x=training_data,
        y=training_targets,
        epochs=5,
        batch_size=2,
        validation_split=0.2,
    )

    print("\n=== Training Complete ===")
    print(f"Final reward: {history.history['reward'][-1]:.4f}")

    # Inspect optimized state
    state_tree = program.get_state_tree()
    optimized_instructions = state_tree["trainable_variables"][
        "recursive_generator"
    ]["state_recursive_generator"]["instructions"]

    print("\n=== Optimized Instructions ===")
    print(optimized_instructions)

    # Use trained program for inference
    test_query = Query(
        question="What innovations are discussed?",
        documents=[
            "Innovation 1: Renewable energy breakthroughs...",
            "Innovation 2: AI-driven drug discovery...",
            "Innovation 3: Quantum computing advances...",
        ],
    )

    result = await program(test_query)
    print("\n=== Inference Result ===")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources}")

    # Save trained program
    await program.save("trained_recursive_analyzer.synalinks")
    print("\n=== Program Saved ===")
    print("Trained program saved to: trained_recursive_analyzer.synalinks")


if __name__ == "__main__":
    asyncio.run(main())
