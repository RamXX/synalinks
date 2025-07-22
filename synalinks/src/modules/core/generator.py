# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import re
from typing import List
from typing import Optional

import jinja2

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import DataModel
from synalinks.src.backend import Instructions
from synalinks.src.backend import Prediction
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib

XML_TAGS_REGEX = re.compile(
    r"<("
    + "|".join([ChatRole.SYSTEM, ChatRole.USER, ChatRole.ASSISTANT])
    + r")\s*(?:[^>]*)>\s*([\s\S]*?)\s*</\1>",
    re.MULTILINE,
)


@synalinks_export("synalinks.default_prompt_template")
def default_prompt_template():
    """Returns the default prompt template.

    Returns:
        (str): The default prompt template.
    """
    return """
<system>
{% if static_system_prompt %}{{ static_system_prompt }}{% endif %}
{% if inputs_schema %}You will be given an input JSON object with the following schema. 
Input JSON Schema:
{{ inputs_schema }}
{% endif %}
{% if outputs_schema %}
Your task is to answer with a JSON object following this output JSON schema.
Output JSON Schema:
{{ outputs_schema }}
{% endif %}
{% if examples %}
### Examples
{% for example in examples %}
Input:
{{ example[0] }}
Output:
{{ example[1] }}
{% endfor %}
{% endif %}
{% if instructions %}
### Instructions:
{% for instruction in instructions %}
 - {{ instruction }}
{% endfor %}
{% endif %}
</system>
{% if inputs %}
<user>
Input:
{{ inputs }}
Output:
</user>
{% endif %}"""


@synalinks_export("synalinks.chat_prompt_template")
def chat_prompt_template():
    """Returns the default chat prompt template.

    Returns:
        (str): The default chat prompt template.
    """
    return """
<system>
{% if instructions %}
### Instructions:
{% for instruction in instructions %}
 - {{ instruction }}
{% endfor %}{% endif %}
</system>
{% for message in inputs.messages %}
{% if message.role == "assistant" %}
<assistant>
{{ message.content }}
</assistant>
{% elif message.role == "user" %}
<user>
{{ message.content }}
</user>
{% else %}{% endif %}
{% endfor %}
"""


class GeneratorState(DataModel):
    """The generator variables."""

    prompt_template: Optional[str] = None
    static_system_prompt: Optional[str] = None
    examples: List[Prediction] = []
    predictions: List[Prediction] = []
    instructions: Optional[Instructions] = None
    instructions_candidates: List[Instructions] = []


@synalinks_export(["synalinks.modules.Generator", "synalinks.Generator"])
class Generator(Module):
    """
    Use a `LanguageModel` to generate a data model from an arbitrary input data model.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():

        class Query(DataModel):
            query: str = synalinks.Field(
                description="The user query",
            )

        class AnswerWithCritique(synalinks.DataModel):
            thinking: str = synalinks.Field(
                description="Your step by step thinking",
            )
            critique: str = synalinks.Field(
                description="The critique of the above thinking",
            )
            answer: str = synalinks.Field(
                description="The correct answer",
            )

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        x0 = synalinks.Input(data_model=Query)
        x1 = await synalinks.Generator(
            data_model=AnswerWithCritique,
            language_model=language_model,
        )(x0)

        program = synalinks.Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought_with_critique",
            description="Useful to answer step by step and evaluate your answer",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Args:
        schema (dict): The target JSON schema.
            If not provided use the `data_model` to infer it.
        data_model (DataModel | SymbolicDataModel | JsonDataModel): The target data
            model for structured output.
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The jinja2 prompt template.
        static_system_prompt (str): A static system prompt that **do not** evolve
            during training. This prompt allow the user to provide additional
            information that won't be changed during training. Allowing to cache
            it and reduce inference costs.
        examples (list): The default list of examples, the examples
            are a list of tuples containing input/output JSON pairs.
        instructions (list): The default instructions being a list of string containing
            additional instructions for the language model.
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the prompt (Default to False).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the prompt (Default to False).
        return_inputs (bool): Optional. Whether or not to concatenate the inputs to
            the outputs (Default to False).
        streaming (str): Optional. If true stream the LM response, enabled only if
            `schema` is `None` and only during inference (not during training).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        schema=None,
        data_model=None,
        language_model=None,
        prompt_template=None,
        static_system_prompt=None,
        examples=None,
        instructions=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        return_inputs=False,
        streaming=False,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema
        self.language_model = language_model
        if not prompt_template:
            prompt_template = default_prompt_template()
        self.prompt_template = prompt_template
        self.static_system_prompt = static_system_prompt
        if not examples:
            examples = []
        self.examples = examples
        if not instructions:
            instructions = []
        self.instructions = instructions

        self.return_inputs = return_inputs
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        if schema and streaming:
            streaming = False
        self.streaming = streaming

        predictions = [
            Prediction(
                inputs=example[0],
                outputs=example[1],
            )
            for example in examples
        ]

        self.state = self.add_variable(
            initializer=GeneratorState(
                static_system_prompt=static_system_prompt,
                prompt_template=prompt_template,
                examples=predictions,
                predictions=predictions,
                instructions=Instructions(instructions=instructions),
            ).get_json(),
            data_model=GeneratorState,
            name=self.name + "_state",
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        msgs = ChatMessages()
        msgs.messages = self.format_messages(inputs)
        if self.streaming and not training:
            streaming = True
        else:
            streaming = False
        result = await ops.predict(
            msgs,
            schema=self.schema,
            language_model=self.language_model,
            streaming=streaming,
            name=self.name + "_prediction",
        )
        if streaming:
            return result
        if result:
            if training:
                self.state.get("predictions").append(
                    Prediction(
                        inputs=inputs.get_json(),
                        outputs=result.get_json(),
                    ).get_json()
                )
            if self.return_inputs:
                return await ops.concat(
                    inputs,
                    result,
                    name=self.name + "_with_inputs",
                )
            else:
                return result
        return None

    async def compute_output_spec(self, inputs, training=False):
        if self.schema:
            if self.return_inputs:
                return await ops.concat(
                    inputs,
                    SymbolicDataModel(
                        schema=self.schema,
                        name=self.name,
                    ),
                    name=self.name + "_with_inputs",
                )
            else:
                return SymbolicDataModel(
                    schema=self.schema,
                    name=self.name,
                )
        else:
            if self.return_inputs:
                return await ops.concat(
                    inputs,
                    SymbolicDataModel(
                        schema=ChatMessage.get_schema(),
                        name=self.name,
                    ),
                    name=self.name + "_with_inputs",
                )
            else:
                return SymbolicDataModel(
                    schema=ChatMessage.get_schema(),
                    name=self.name,
                )

    def format_messages(self, inputs=None):
        template = jinja2.Template(self.state.get("prompt_template"))
        rendered_prompt = template.render(
            static_system_prompt=self.static_system_prompt,
            inputs_schema=inputs.get_schema() if self.use_inputs_schema else None,
            outputs_schema=self.schema if self.use_outputs_schema else None,
            examples=[
                (pred.get("inputs"), pred.get("outputs"))
                for pred in self.state.get("examples")
            ],
            instructions=self.state.get("instructions").get("instructions"),
            inputs=inputs.get_json() if inputs else None,
        )
        matches = XML_TAGS_REGEX.findall(rendered_prompt)
        extracted_tags = [(match[0], match[1].strip()) for match in matches]
        messages = []
        for message in extracted_tags:
            role, content = message
            if content:
                messages.append(ChatMessage(role=role, content=content))
        return messages

    def get_config(self):
        config = {
            "schema": self.schema,
            "prompt_template": self.prompt_template,
            "static_system_prompt": self.static_system_prompt,
            "examples": self.examples,
            "instructions": self.instructions,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "return_inputs": self.return_inputs,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            )
        }
        return {**config, **language_model_config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model"),
        )
        return cls(language_model=language_model, **config)
