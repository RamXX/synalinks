"""RecursiveChainOfThought module combining RLM with chain-of-thought reasoning."""

from typing import Optional
from typing import Union

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.core.chunking_strategy import ChunkingStrategy
from synalinks.src.modules.rlm.core.recursive_generator import RecursiveGenerator
from synalinks.src.saving import serialization_lib


class Thinking(DataModel):
    """Thinking step data model."""

    thinking: str = Field(description="Your step by step thinking")


def default_cot_rlm_instructions(data_model_fields):
    """Default CoT-RLM instructions for structured output with step-by-step reasoning."""
    return f"""
You are a recursive language model that can execute Python code and make
sub-LLM calls. You are tasked with answering queries using step-by-step reasoning.

Your task is to analyze the input and produce a JSON output with thinking steps
followed by the final answer with these keys: {data_model_fields}

## Approach

Use the REPL environment combined with deliberate reasoning to:
1. Think step-by-step about the problem in the thinking fields
2. Break down complex tasks into smaller steps using code
3. Execute Python code to process data
4. Make recursive llm_query() calls for sub-problems
5. Verify your reasoning through sub-calls when needed
6. Build up your final answer incrementally

When you have the final answer, use FINAL_VAR(variable_name) to return
the structured result with both thinking steps and the final answer.
""".strip()


@synalinks_export(
    [
        "synalinks.modules.RecursiveChainOfThought",
        "synalinks.RecursiveChainOfThought",
    ]
)
class RecursiveChainOfThought(RecursiveGenerator):
    """Recursive Language Model generator with chain-of-thought reasoning.

    Combines RLM's recursive REPL execution with chain-of-thought prompting
    by prepending thinking fields to the output schema. The LLM is encouraged
    to think step-by-step before providing the final answer, while also
    being able to execute code and make recursive LLM calls for complex tasks.

    **Inherits Multi-Model Architecture**:
    Like RecursiveGenerator, supports separate language_model (root) and
    sub_language_model (recursive calls) for cost optimization.

    The parameter `k` specifies the number of thinking fields to add.

    Args:
        schema (dict): Target JSON schema. If not provided, use data_model.
        data_model (DataModel): Target data model for structured output.
        language_model (LanguageModel): Root language model for orchestration.
        sub_language_model (LanguageModel): Language model for recursive sub-calls.
            If not provided, uses language_model for both.
        instructions (str): System prompt instructions. Trainable via Variable.
        seed_instructions (list): Seed instructions for optimization.
        examples (list): Few-shot examples as (input, output) tuples. Trainable.
        max_iterations (int): Maximum RLM loop iterations (default 30).
        max_depth (int): Maximum recursion depth (default 1).
        k (int): Number of thinking fields to prepend (default 1).
        temperature (float): LLM temperature (default 0.0).
        chunking_strategy (ChunkingStrategy | str): Optional chunking for large inputs.
        enable_trajectory_logging (bool): Enable detailed trajectory logging (default False).
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether module variables are trainable.

    Example:
        >>> import synalinks
        >>> import asyncio
        >>>
        >>> class Query(synalinks.DataModel):
        ...     query: str = synalinks.Field(description="The user query")
        ...     documents: list[str] = synalinks.Field(description="Large document set")
        >>>
        >>> class Answer(synalinks.DataModel):
        ...     answer: str = synalinks.Field(description="The answer")
        ...     confidence: float = synalinks.Field(description="Confidence 0-1")
        >>>
        >>> async def main():
        ...     # Multi-model setup - inherits from RecursiveGenerator
        ...     lm_root = synalinks.LanguageModel(model="zai/glm-4.7")
        ...     lm_sub = synalinks.LanguageModel(model="groq/openai/gpt-oss-20b")
        ...
        ...     gen = synalinks.RecursiveChainOfThought(
        ...         data_model=Answer,
        ...         language_model=lm_root,
        ...         sub_language_model=lm_sub,
        ...         k=3,  # 3 thinking steps
        ...     )
        ...
        ...     query = Query(
        ...         query="Summarize key findings",
        ...         documents=["doc1...", "doc2...", "doc3..."]
        ...     )
        ...     result = await gen(query.to_json_data_model())
        ...     print(result)
        >>>
        >>> asyncio.run(main())

    References:
        - Chain-of-Thought: https://arxiv.org/abs/2201.11903
        - RLM: https://github.com/recursionpharma/rlm
    """

    def __init__(
        self,
        schema=None,
        data_model=None,
        language_model: Optional[LanguageModel] = None,
        sub_language_model: Optional[LanguageModel] = None,
        instructions=None,
        seed_instructions=None,
        examples=None,
        max_iterations: int = 30,
        max_depth: int = 1,
        k: int = 1,
        temperature: float = 0.0,
        chunking_strategy: Optional[Union[str, ChunkingStrategy]] = None,
        enable_trajectory_logging: bool = False,
        name=None,
        description=None,
        trainable=True,
    ):
        # Store original schema and k before augmentation
        if not schema and data_model:
            schema = data_model.get_schema()
        self._original_schema = schema
        self.k = k

        # Build thinking-augmented schema following ChainOfThought pattern
        thinking_data_model = Thinking
        if k > 1:
            for _ in range(k - 1):
                thinking_data_model = thinking_data_model + Thinking

        augmented_data_model = thinking_data_model + SymbolicDataModel(schema=schema)

        # Enhanced instructions for CoT if not provided
        if not instructions and schema:
            data_model_keys = list(augmented_data_model.get_schema()["properties"].keys())
            instructions = default_cot_rlm_instructions(data_model_keys)

        # Initialize RecursiveGenerator with augmented schema
        super().__init__(
            schema=None,  # Pass via data_model instead
            data_model=augmented_data_model,
            language_model=language_model,
            sub_language_model=sub_language_model,
            instructions=instructions,
            seed_instructions=seed_instructions,
            examples=examples,
            max_iterations=max_iterations,
            max_depth=max_depth,
            temperature=temperature,
            chunking_strategy=chunking_strategy,
            enable_trajectory_logging=enable_trajectory_logging,
            name=name,
            description=description,
            trainable=trainable,
        )

    def get_config(self):
        """Serialize module configuration.

        Returns:
            dict: Configuration dict including k parameter
        """
        config = super().get_config()
        # Add k parameter and restore original schema
        config["k"] = self.k
        config["schema"] = self._original_schema
        return config

    @classmethod
    def from_config(cls, config):
        """Deserialize module from config.

        Args:
            config (dict): Configuration dict

        Returns:
            RecursiveChainOfThought: RecursiveChainOfThought instance
        """
        # Extract k before passing to parent
        k = config.pop("k", 1)

        # Deserialize language models
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        sub_language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("sub_language_model")
        )

        # Deserialize chunking_strategy if present
        chunking_strategy = None
        if "chunking_strategy" in config:
            chunking_strategy = serialization_lib.deserialize_synalinks_object(
                config.pop("chunking_strategy")
            )

        # Extract enable_trajectory_logging if present
        enable_trajectory_logging = config.pop("enable_trajectory_logging", False)

        return cls(
            language_model=language_model,
            sub_language_model=sub_language_model,
            k=k,
            chunking_strategy=chunking_strategy,
            enable_trajectory_logging=enable_trajectory_logging,
            **config,
        )
