"""RecursiveGenerator module for RLM integration."""

from typing import Optional
from typing import Union

from synalinks.src.api_export import synalinks_export
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.module import Module
from synalinks.src.modules.rlm.core.chunking_strategy import ChunkingStrategy
from synalinks.src.modules.rlm.core.chunking_strategy import get_chunking_strategy
from synalinks.src.modules.rlm.prompts.templates import get_prompt_template


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
        max_depth: Maximum recursion depth for llm_query calls (default: 1)
        chunking_strategy: Optional chunking strategy for large inputs. Can be a
            ChunkingStrategy instance or a string ('uniform', 'keyword', 'semantic')
        prompt_template: Optional Jinja2 prompt template. If None, automatically
            selects template based on language_model.model prefix (e.g., 'zai', 'groq')
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
        max_depth: int = 1,
        chunking_strategy: Optional[Union[str, ChunkingStrategy]] = None,
        prompt_template: Optional[str] = None,
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
        # Default sub_language_model to language_model for single-model usage
        self.sub_language_model = (
            sub_language_model if sub_language_model is not None else language_model
        )
        self.max_iterations = max_iterations
        self.max_depth = max_depth

        # Initialize chunking strategy
        self.chunking_strategy = None
        if chunking_strategy is not None:
            if isinstance(chunking_strategy, str):
                self.chunking_strategy = get_chunking_strategy(chunking_strategy)
            else:
                self.chunking_strategy = chunking_strategy

        # Initialize prompt template (auto-detect if not provided)
        if prompt_template is None:
            # Auto-detect based on language_model.model prefix
            self.prompt_template = get_prompt_template(language_model.model)
        else:
            self.prompt_template = prompt_template

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
        from synalinks.src.saving import serialization_lib

        base_config = super().get_config()
        config = {
            "data_model": self.data_model,
            "language_model": self.language_model,
            "sub_language_model": self.sub_language_model,
            "max_iterations": self.max_iterations,
            "max_depth": self.max_depth,
            "prompt_template": self.prompt_template,
        }

        # Serialize chunking_strategy if present
        if self.chunking_strategy is not None:
            config["chunking_strategy"] = serialization_lib.serialize_synalinks_object(
                self.chunking_strategy
            )

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Deserialize from config."""
        from synalinks.src.saving import serialization_lib

        # Deserialize chunking_strategy if present
        if "chunking_strategy" in config:
            config["chunking_strategy"] = serialization_lib.deserialize_synalinks_object(
                config["chunking_strategy"]
            )

        return cls(**config)
