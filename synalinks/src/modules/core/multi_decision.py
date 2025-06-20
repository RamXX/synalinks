# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json
import copy
from synalinks.src import ops
from typing import Set, List
from enum import Enum
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import dynamic_enum 
# from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_enum_list
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


#--------------------------------------------------------------------------------------------------
def dynamic_enum_list(schema, prop_to_update, labels, parent_schema=None, description=None):
    """Update a schema with dynamic Enum list for array properties.

    This function takes a schema with an array property and constrains the items
    in that array to be from a specific enum of labels.

    Args:
        schema (dict): The schema to update (should contain an array property).
        prop_to_update (str): The array property to update with enum constraints.
        labels (list): The list of labels (strings) for the enum.
        parent_schema (dict, optional): An optional parent schema to use as the base.
        description (str, optional): An optional description for the enum.

    Returns:
        dict: The updated schema with the enum applied to the array items.
    """
    schema = copy.deepcopy(schema)
    
    # Ensure $defs is at the beginning of the schema
    if schema.get("$defs"):
        schema = {"$defs": schema.pop("$defs"), **schema}
    else:
        schema = {"$defs": {}, **schema}
    
    if parent_schema:
        parent_schema = copy.deepcopy(parent_schema)
    
    # Create enum title (capitalize and remove underscores)
    enum_title = prop_to_update.title().replace("_", "").rstrip("s")  # Remove trailing 's' for singular form
    if enum_title.endswith("Choice"):
        enum_title = enum_title
    else:
        enum_title = enum_title.rstrip("s") if enum_title.endswith("s") else enum_title
        if not enum_title.endswith("Choice"):
            enum_title = "Choice"  # Default to "Choice" for consistency
    
    # Create the enum definition
    if description:
        enum_definition = {
            "enum": labels,
            "description": description,
            "title": enum_title,
            "type": "string",
        }
    else:
        enum_definition = {
            "enum": labels,
            "title": enum_title,
            "type": "string",
        }
    
    # Add enum to $defs
    target_schema = parent_schema if parent_schema else schema
    target_schema["$defs"].update({enum_title: enum_definition})
    
    # Update the array property to reference the enum
    if "properties" in schema and prop_to_update in schema["properties"]:
        # Update existing property
        schema["properties"][prop_to_update].update({
            "items": {
                "$ref": f"#/$defs/{enum_title}"
            },
            "type": "array",
            "uniqueItems": True  # Ensure unique items as shown in desired output
        })
        
        # Preserve existing description and title if they exist
        if "description" not in schema["properties"][prop_to_update]:
            schema["properties"][prop_to_update]["description"] = "The labels choosed."
        if "title" not in schema["properties"][prop_to_update]:
            schema["properties"][prop_to_update]["title"] = prop_to_update.title()
    
    return parent_schema if parent_schema else schema
#-------------------------------------------------------------------------------------------------

class Question(DataModel):
    question: str = Field(description="The question to ask yourself.")


# class DecisionAnswer(DataModel):
#     thinking: str = Field(
#         description="Your step by step thinking to choose the correct label."
#     )
#     choice: str = Field(description="The label choosed.")


# print("------------------Singel_DecisionAnswer_Schema--------------------------------------------------")
# schema = DecisionAnswer.get_schema()
# print(json.dumps(schema, indent=2))


class MultipleDecisionAnswer(DataModel):
    thinking: str = Field(
        description="Your step by step thinking to choose the correct label."
    )
    choices: List[str] = Field(description="The labels choosed.")

print("------------------Multiple_DecisionAnswer_Schema--------------------------------------------------")
multischema = MultipleDecisionAnswer.get_schema()
print(json.dumps(multischema, indent=2))


# print("------------------singel_dynamic_enum_schema--------------------------------------------------")
# enum_schema = dynamic_enum(schema, prop_to_update="choice", labels=["t1", "t2"])
# print(json.dumps(enum_schema, indent=2))

print("------------------dynamic_enum_list_schema--------------------------------------------------")
enum_schema = dynamic_enum_list(multischema, prop_to_update="choice", labels=["t1", "t2"])
print(json.dumps(enum_schema, indent=2))




@synalinks_export(["synalinks.modules.Decision", "synalinks.Decision"])
class Decision(Module):
    """Perform a decision on the given input based on a question and a list of labels.

    This module dynamically create an `Enum` schema based on the given labels and
    use it to generate a possible answer using structured output.

    This ensure that the LM answer is **always** one of the provided labels.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        x0 = synalinks.Input(data_model=synalinks.ChatMessages)
        x1 = await synalinks.Decision(
            question="What is the danger level of the discussion?",
            labels=["low", "medium", "high"],
            language_model=language_model,
        )(x0)

        program = synalinks.Program(
            inputs=x0,
            outputs=x1,
            name="discussion_danger_assessment",
            description="This program assesses the level of danger in a discussion.",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    You can view this module, as performing a single label classification on the input.

    Args:
        question (str): The question to ask.
        labels (list): The list of labels to choose from (strings).
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The default jinja2 prompt template
            to use (see `Generator`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        instructions (list): The default instructions to use (see `Generator`).
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the prompt (Default to False) (see `Generator`).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the prompt (Default to False) (see `Generator`).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        question=None,
        labels=None,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        if not question:
            raise ValueError("The `question` argument must be provided.")
        if not labels:
            raise ValueError("The `labels` argument must be provided.")
        if not isinstance(labels, list):
            raise ValueError("The `labels` parameter must be a list of string.")
        # schema = dynamic_enum(DecisionAnswer.get_schema(), "choice", labels)
        # self.schema = schema
        schema = dynamic_enum_list(MultipleDecisionAnswer.get_schema(), "choices", labels)
        self.schema = schema
        self.question = question
        self.labels = labels
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.decision = Generator(
            schema=schema,
            language_model=language_model,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            use_inputs_schema=use_inputs_schema,
            use_outputs_schema=use_outputs_schema,
            name=self.name + "_generator",
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        inputs = await ops.concat(
            inputs,
            Question(question=self.question),
            name=self.name + "_inputs_with_question",
        )
        result = await self.decision(inputs, training=training)
        return result

    def get_config(self):
        config = {
            "question": self.question,
            "labels": self.labels,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            )
        }
        return {**config, **language_model_config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(language_model=language_model, **config)
