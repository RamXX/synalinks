"""RecursiveGenerator module for RLM integration."""

from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.module import Module


@synalinks_export(
    ["synalinks.modules.RecursiveGenerator", "synalinks.RecursiveGenerator"]
)
class RecursiveGenerator(Module):
    """Generator with RLM capabilities for code execution and recursive calls.

    Extends Generator pattern with ability to:
    - Execute Python code in sandboxed REPL
    - Make recursive sub-LM calls for decomposition
    - Parse FINAL()/FINAL_VAR() termination patterns

    This is a walking skeleton implementation proving the integration works.

    Args:
        data_model: Target DataModel for structured output
        language_model: Root LanguageModel for orchestration
        sub_language_model: Optional sub-LM for recursive calls (cost optimization)
        max_iterations: Maximum REPL iterations before termination
        name: Module name
        description: Module description
        trainable: Whether module variables are trainable

    Example:
        >>> import synalinks
        >>> import asyncio
        >>>
        >>> async def main():
        ...     class Query(synalinks.DataModel):
        ...         query: str = synalinks.Field(description="User query")
        ...
        ...     class Answer(synalinks.DataModel):
        ...         answer: str = synalinks.Field(description="The answer")
        ...
        ...     lm_root = synalinks.LanguageModel(model="zai/glm-4.7")
        ...     lm_sub = synalinks.LanguageModel(model="groq/openai/gpt-oss-20b")
        ...
        ...     gen = synalinks.RecursiveGenerator(
        ...         data_model=Answer,
        ...         language_model=lm_root,
        ...         sub_language_model=lm_sub,
        ...     )
        ...
        ...     query = Query(query="What is 2+2?")
        ...     result = await gen(query)
        ...
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        data_model=None,
        language_model: Optional[LanguageModel] = None,
        sub_language_model: Optional[LanguageModel] = None,
        max_iterations: int = 10,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )

        if not language_model:
            raise ValueError("language_model parameter is required")

        self.data_model = data_model
        self.language_model = language_model
        self.sub_language_model = sub_language_model
        self.max_iterations = max_iterations

        # Schema for structured output
        self.schema = None
        if data_model:
            self.schema = data_model.get_schema()

    async def call(self, inputs, training=False):
        """Execute recursive generation workflow.

        This is a minimal implementation that proves the integration works.
        Full implementation will include REPL execution, parsing, and recursion.

        Args:
            inputs: Input DataModel
            training: Whether in training mode

        Returns:
            Output DataModel matching schema
        """
        # Walking skeleton: Just return a mock result
        # Full implementation will:
        # 1. Create LMHandler with registered clients
        # 2. Create LocalREPL with llm_query injection
        # 3. Enter iteration loop:
        #    - Call root LM to get response
        #    - Parse for code blocks
        #    - Execute code in REPL
        #    - Check for FINAL pattern
        #    - Format results and continue
        # 4. Extract final answer and return

        # Walking skeleton: Just pass through a minimal result
        # Full implementation will do actual RLM execution
        from synalinks.src.backend import ChatMessage

        return ChatMessage(role="assistant", content="Walking skeleton response")

    def get_config(self):
        """Get serialization config."""
        base_config = super().get_config()
        config = {
            "data_model": self.data_model,
            "language_model": self.language_model,
            "sub_language_model": self.sub_language_model,
            "max_iterations": self.max_iterations,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Deserialize from config."""
        return cls(**config)
