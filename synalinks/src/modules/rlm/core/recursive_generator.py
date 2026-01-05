"""RecursiveGenerator module for RLM integration."""

import json
from typing import Optional
from typing import Union

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatRole
from synalinks.src.backend import Instructions
from synalinks.src.backend import Prediction
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.module import Module
from synalinks.src.modules.rlm.clients.synalinks_adapter import SynalinksLMClient
from synalinks.src.modules.rlm.core.chunking_strategy import ChunkingStrategy
from synalinks.src.modules.rlm.core.chunking_strategy import get_chunking_strategy
from synalinks.src.modules.rlm.core.lm_handler import LMHandler
from synalinks.src.modules.rlm.core.local_repl import LocalREPL
from synalinks.src.modules.rlm.utils.parsing import find_code_blocks
from synalinks.src.modules.rlm.utils.parsing import find_final_answer
from synalinks.src.modules.rlm.utils.parsing import format_execution_result
from synalinks.src.saving import serialization_lib


def default_rlm_instructions(data_model_fields):
    """Default RLM instructions for structured output."""
    return f"""
You are a recursive language model that can execute Python code and make
sub-LLM calls.

Your task is to analyze the input and produce a JSON output with the
following keys: {data_model_fields}

Use the REPL environment to:
1. Break down complex tasks into smaller steps
2. Execute Python code to process data
3. Make recursive llm_query() calls for sub-problems
4. Build up your final answer incrementally

When you have the final answer, use FINAL_VAR(variable_name) to return
the structured result.
""".strip()


@synalinks_export(
    ["synalinks.modules.RecursiveGenerator", "synalinks.RecursiveGenerator"]
)
class RecursiveGenerator(Module):
    """Recursive Language Model generator using REPL code execution.

    Enables LLMs to overcome context window limitations by executing Python
    code in a sandboxed REPL environment, making recursive sub-LLM calls
    to process large contexts through programmatic decomposition.

    **Multi-Model Architecture (Paper Insight)**:
    Use separate models for root (orchestration) and sub (recursive) calls
    for cost optimization. Paper shows this as 'GPT-5 with medium reasoning
    (using GPT-5-mini for recursive calls)'.

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
        temperature (float): LLM temperature (default 0.0).
        chunking_strategy (ChunkingStrategy | str): Optional chunking for large inputs.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether module variables are trainable.

    Example:
        >>> import synalinks
        >>> import asyncio
        >>>
        >>> class Query(synalinks.DataModel):
        ...     query: str = synalinks.Field(description="User query")
        ...     context: str = synalinks.Field(description="Context")
        >>>
        >>> class Answer(synalinks.DataModel):
        ...     answer: str = synalinks.Field(description="The answer")
        >>>
        >>> async def main():
        ...     # Multi-model setup for cost optimization
        ...     lm_root = synalinks.LanguageModel(model="zai/glm-4.7")
        ...     lm_sub = synalinks.LanguageModel(model="groq/openai/gpt-oss-20b")
        ...
        ...     gen = synalinks.RecursiveGenerator(
        ...         data_model=Answer,
        ...         language_model=lm_root,
        ...         sub_language_model=lm_sub,
        ...         max_iterations=30,
        ...     )
        ...
        ...     query = Query(query="What is 2+2?", context="Math problem")
        ...     result = await gen(query.to_json_data_model())
        ...     print(result)
        >>>
        >>> asyncio.run(main())

    References:
        - RLM: Recursive Language Models (https://github.com/recursionpharma/rlm)
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
        temperature: float = 0.0,
        chunking_strategy: Optional[Union[str, ChunkingStrategy]] = None,
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
            raise ValueError("You must provide `language_model` parameter.")

        # Schema for structured output
        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema

        self.language_model = language_model
        # Paper insight: Use cheaper model for sub-calls
        self.sub_language_model = sub_language_model or language_model

        # Default instructions based on schema
        if not instructions and self.schema:
            data_model_keys = list(self.schema["properties"].keys())
            instructions = default_rlm_instructions(data_model_keys)
        self.instructions = instructions

        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.temperature = temperature

        # Examples for few-shot learning
        if not examples:
            examples = []
        self.examples = examples

        # Seed instructions for optimization
        if not seed_instructions:
            seed_instructions = []
        self.seed_instructions = seed_instructions

        # Chunking strategy
        self.chunking_strategy = None
        if chunking_strategy is not None:
            if isinstance(chunking_strategy, str):
                self.chunking_strategy = get_chunking_strategy(chunking_strategy)
            else:
                self.chunking_strategy = chunking_strategy

        # Create trainable state variable following Generator pattern
        predictions = [
            Prediction(
                inputs=example[0],
                outputs=example[1],
                reward=None,
            ).get_json()
            for example in examples
        ]

        seed_candidates = [{"instructions": seed} for seed in self.seed_instructions]

        self.state = self.add_variable(
            initializer=Instructions(
                instructions=instructions,
                examples=predictions,
                seed_candidates=seed_candidates,
            ).get_json(),
            data_model=Instructions,
            name="state_" + self.name,
        )

    def _format_system_prompt(self, inputs_schema=None):
        """Format system prompt with instructions and schema.

        Args:
            inputs_schema (str): Optional input schema to include in prompt

        Returns:
            ChatMessage: Formatted system message
        """
        # Build prompt sections
        sections = []

        # Instructions
        instructions = self.state.get("instructions")
        if instructions:
            sections.append(instructions)

        # Output schema
        if self.schema:
            schema_json = json.dumps(self.schema, indent=2)
            sections.append(f"\n# Expected Output Schema\n{schema_json}")

        # Examples
        examples = self.state.get("examples") or []
        if examples:
            sections.append("\n# Examples")
            for i, example in enumerate(examples):
                sections.append(f"\nExample {i+1}:")
                sections.append(f"Input: {json.dumps(example.get('inputs'), indent=2)}")
                sections.append(f"Output: {json.dumps(example.get('outputs'), indent=2)}")

        content = "\n".join(sections)
        return ChatMessage(role=ChatRole.SYSTEM, content=content)

    def _format_user_prompt(self, inputs, iteration: int = 0):
        """Format user prompt with input data.

        Args:
            inputs: Input data model
            iteration (int): Current iteration number

        Returns:
            ChatMessage: User message
        """
        content = f"# Input\n{json.dumps(inputs.get_json(), indent=2)}"
        if iteration > 0:
            content += f"\n\n(Iteration {iteration})"
        return ChatMessage(role=ChatRole.USER, content=content)

    async def call(self, inputs, training=False):
        """Execute RLM loop to generate structured output.

        1. Create adapters for root and sub LMs
        2. Start LMHandler with both clients registered
        3. Initialize REPL with input context
        4. Run RLM loop: prompt -> code -> execute -> FINAL extraction
        5. Parse FINAL_VAR to structured output
        6. Cleanup resources

        Args:
            inputs: Input DataModel
            training (bool): Whether in training mode

        Returns:
            Output DataModel matching schema
        """
        if not inputs:
            return None

        # Create LM clients - separate for root and sub
        root_client = SynalinksLMClient(self.language_model)
        sub_client = SynalinksLMClient(self.sub_language_model)

        # Create and start LMHandler
        lm_handler = LMHandler()
        try:
            # Register root client as default
            lm_handler.register_client(
                self.language_model.model, root_client, is_default=True
            )

            # Register sub-client for recursive calls if different
            if self.sub_language_model.model != self.language_model.model:
                lm_handler.register_client(self.sub_language_model.model, sub_client)

            lm_handler.start()

            # Create REPL with llm_query injected
            llm_query_fn = lm_handler.create_llm_query_fn(
                client_name=self.sub_language_model.model,
                current_depth=0,
                max_depth=self.max_depth,
            )

            repl = LocalREPL(
                llm_query_fn=llm_query_fn,
                default_sub_model=self.sub_language_model.model,
                current_depth=0,
                max_depth=self.max_depth,
            )

            try:
                # Load input context into REPL
                context = inputs.get_json()
                repl.load_context({"input_data": context})

                # Build initial message history
                system_msg = self._format_system_prompt()
                user_msg = self._format_user_prompt(inputs)
                message_history = [system_msg, user_msg]

                # RLM loop
                for i in range(self.max_iterations):
                    # Convert messages to dict format for client
                    msg_dicts = [
                        {"role": msg.role, "content": msg.content}
                        for msg in message_history
                    ]

                    # Get LLM response using root model
                    response = await root_client.acompletion(msg_dicts)

                    # Find and execute code blocks
                    code_blocks = find_code_blocks(response)
                    exec_results = []
                    for code in code_blocks:
                        result = repl.execute(code)
                        exec_results.append(result)

                        # Check if FINAL_VAR was set during execution
                        if result.final_answer is not None:
                            output = self._parse_output(result.final_answer)
                            if training and output:
                                self._track_prediction(inputs, output)
                            return output

                    # Check for FINAL() or FINAL_VAR() pattern in response
                    final = find_final_answer(response)
                    if final:
                        answer_type, content = final
                        if answer_type == "FINAL_VAR":
                            # Get variable from REPL
                            try:
                                content = repl.get_variable(content)
                            except KeyError:
                                # Variable not found, use content as-is
                                pass

                        output = self._parse_output(content)
                        if training and output:
                            self._track_prediction(inputs, output)
                        return output

                    # Add assistant response to history
                    message_history.append(
                        ChatMessage(role=ChatRole.ASSISTANT, content=response)
                    )

                    # Add execution results to history
                    if exec_results:
                        exec_summaries = [
                            format_execution_result(result) for result in exec_results
                        ]
                        message_history.append(
                            ChatMessage(
                                role=ChatRole.USER,
                                content="\n\n".join(exec_summaries),
                            )
                        )

                # Max iterations reached - try to extract final answer
                return await self._default_answer(
                    message_history, root_client, inputs, training
                )

            finally:
                repl.cleanup()

        finally:
            lm_handler.stop()

    def _parse_output(self, content):
        """Parse string content to output DataModel.

        Args:
            content: String or dict content to parse

        Returns:
            Parsed DataModel or raw content
        """
        try:
            if self.schema:
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content
                return SymbolicDataModel(schema=self.schema).from_json(data)
            return content
        except (json.JSONDecodeError, Exception):
            return content

    async def _default_answer(self, message_history, client, inputs, training):
        """Generate answer when max iterations reached.

        Args:
            message_history (list): Message history so far
            client (SynalinksLMClient): LM client to use
            inputs: Original inputs
            training (bool): Whether in training mode

        Returns:
            Parsed output or None
        """
        # Ask for final answer
        message_history.append(
            ChatMessage(
                role=ChatRole.USER,
                content="Max iterations reached. Please provide a final answer now.",
            )
        )

        msg_dicts = [
            {"role": msg.role, "content": msg.content} for msg in message_history
        ]
        response = await client.acompletion(msg_dicts)

        output = self._parse_output(response)
        if training and output:
            self._track_prediction(inputs, output)
        return output

    def _track_prediction(self, inputs, output):
        """Track prediction for training.

        Args:
            inputs: Input DataModel
            output: Output DataModel
        """
        predictions = self.state.get("current_predictions")
        predictions.append(
            {
                "inputs": inputs.get_json(),
                "outputs": output.get_json() if hasattr(output, "get_json") else output,
                "reward": None,
            }
        )

    async def compute_output_spec(self, inputs, training=False):
        """Return output schema for graph building.

        Args:
            inputs: Input specification
            training (bool): Whether in training mode

        Returns:
            SymbolicDataModel or None
        """
        if self.schema:
            return SymbolicDataModel(schema=self.schema, name=self.name)
        return None

    def get_config(self):
        """Serialize module configuration.

        Returns:
            dict: Configuration dict
        """
        base_config = super().get_config()
        config = {
            "schema": self.schema,
            "instructions": self.instructions,
            "seed_instructions": self.seed_instructions,
            "examples": self.examples,
            "max_iterations": self.max_iterations,
            "max_depth": self.max_depth,
            "temperature": self.temperature,
        }

        # Serialize language models
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            ),
            "sub_language_model": serialization_lib.serialize_synalinks_object(
                self.sub_language_model
            ),
        }

        # Serialize chunking_strategy if present
        if self.chunking_strategy is not None:
            config["chunking_strategy"] = serialization_lib.serialize_synalinks_object(
                self.chunking_strategy
            )

        return {**base_config, **config, **language_model_config}

    @classmethod
    def from_config(cls, config):
        """Deserialize module from config.

        Args:
            config (dict): Configuration dict

        Returns:
            RecursiveGenerator: RecursiveGenerator instance
        """
        # Deserialize language models
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        sub_language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("sub_language_model")
        )

        # Deserialize chunking_strategy if present
        if "chunking_strategy" in config:
            config["chunking_strategy"] = serialization_lib.deserialize_synalinks_object(
                config["chunking_strategy"]
            )

        return cls(
            language_model=language_model,
            sub_language_model=sub_language_model,
            **config,
        )
