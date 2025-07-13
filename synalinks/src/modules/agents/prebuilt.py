# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import copy
from typing import List

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel, Field, JsonDataModel
from synalinks.src.modules.core.action import Action
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.module import Module
from synalinks.src.saving.serialization_lib import deserialize_synalinks_object, serialize_synalinks_object
from synalinks.src.utils.nlp_utils import to_singular_property
from synalinks.src.utils.tool_utils import Tool, toolkit_to_static_prompt


class ToolChoice(DataModel):
    tool: str
    purpose: str = Field(
        description="A clear, specific explanation of what the tool should accomplish."
    )


class ToolDecision(DataModel):
    reasoning: str = Field(
        description="A step-by-step analysis of the current state, what has been done, and what should be done next."
    )
    choices: List[ToolChoice] = Field(
        description="The array of tool choices to run in parallel with their specific purpose."
    )


class Question(DataModel):
    toolkit: str = Field(description="The toolkit of available tools")


class Purpose(DataModel):
    purpose: str = Field(description="The purpose of the running action")


def dynamic_enum_on_nested_property(schema, property, labels, description=None):
    """Update a JSON schema with a dynamic enum on a nested property.
 
    Args:
        schema (dict): The schema to update.
        property (str): The nested, slash-separated property path
        labels (list): The array of enum options (strings).
        description (str, optional): The optional enum description

    Returns:
        dict: The updated JSON schema with the enum applied to the nested property.
    """
    schema = copy.deepcopy(schema)
    
    if schema.get("$defs"):
        schema = {"$defs": schema.pop("$defs"), **schema}
    else:
        schema = {"$defs": {}, **schema}

    property = property.split("/")
    klass = to_singular_property(property[-1]).title().replace("_", " ")
    
    definition = {
        "enum": labels,
        "title": klass,
        "type": "string",
    }
    
    if description:
        definition["description"] = description
    
    schema["$defs"].update({klass: definition})

    entity = schema["$defs"]
    
    for part in property[:-1]:
        if part == "items":
            entity = entity.setdefault("items", {})
        elif part == "properties":
            entity = entity.setdefault("properties", {})
        else:
            entity = entity.setdefault(part, {})
    
    entity[property[-1]] = {"$ref": f"#/$defs/{klass}"}
    
    return schema


def get_default_decision_question(toolkit: List[Tool] = None):
    """The default question prompt to make decisions in a ReAct agent."""
    toolkit = toolkit or []
    toolkit = toolkit_to_static_prompt(toolkit)

    question = Question(toolkit=toolkit)

    return question


def get_default_decision_instructions() -> List[str]:
    """The default guiding instructions to make decisions in a ReAct agent."""
    return [
        "Analyze the current state: What do you observe? What do you need to accomplish next? Before taking any action, carefully consider context and all available information.",
        "Reflect on prior steps: Review your previous actions and their outcomes to avoid unnecessary repetition.",
        "Reason methodically: Think step-by-step to determine the most appropriate choices with the available toolkit.",
        "Avoid unnecessary actions: If you already have enough information to complete the user task, return an empty choices array.",
        "Keep subgoals atomic: Each choice must focus on accomplishing exactly one specific task. Break down complex objectives into multiple separate choices rather than creating compound subgoals.",
        "Split complex steps across parallel choices: When facing a multi-part objective, create multiple choices using the same tool with different atomic subgoals that can execute simultaneously without dependencies.",
        "Write self-contained subgoals: Each subgoal must be completely self-explanatory without referencing context, other subgoals, or using pronouns. Include all necessary information explicitly within the subgoal description.",
        "Ensure parallel independence: Make tool choices that can execute concurrently without requiring results from each other. Avoid creating dependencies between parallel subgoals.",
    ]


@synalinks_export(["synalinks.modules.Agent", "synalinks.Agent"])
class Agent(Module):
    """
    ReAct agent as a directed acyclic graph that chooses at each step which tools to run.

    Note:
        - Each function must return a JSON object and be asynchrounous.

    References:
        - [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

    Args:
        schema (dict): The JSON schema to use for the final answer.
            If not provided, it will use the `output_data_model` argument.
        data_model (DataModel | JsonDataModel | SymbolicDataModel): Optional.
            The data model to use for the final answer.
            If None provided, the Agent will return a ChatMessage-like data model.
        toolkit (list): The toolkit of functions for the agent to choose from.
        question (str): Optional. The question to branch on actions at each step.
        language_model (LanguageModel): The language model to use, if provided
            it will ignore `decision_language_model` and `action_language_model` argument.
        decision_language_model (LanguageModel): The language model used for
            decision-making.
        action_language_model (LanguageModel): The language model used for actions.
        prompt_template (str): Optional. The jinja2 prompt template to use
            (See `Generator`).
        examples (list): A default list of examples for decision-making (See `Decision`).
        instructions (list): A default list of instructions for decision-making
            (See `Decision`).
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the decision prompt (Default to False) (see `Decision`).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the decision prompt (Default to False) (see `Decision`).
        return_inputs_with_trajectory (bool): Optional. Whether or not to concatenate the
            inputs along with the agent trajectory to the outputs (Default to False).
        return_inputs_only (bool): Optional. Whether or not to concatenate the inputs
            to the outputs (Default to False).
        max_iterations (int): The maximum number of steps to perform.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        schema=None,
        data_model=None,
        toolkit=None,
        question=None,
        language_model=None,
        decision_language_model=None,
        action_language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        return_type="response",  # I/O, trajectory
        return_inputs_with_trajectory=False,
        return_inputs_only=False,
        max_iterations=10,
        name=None,
        description=None,
        trainable=True,

    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )

        if schema:
            self.schema = schema
        elif data_model:
            self.schema = data_model.get_schema()
        else:
            raise ValueError(
                "You must set either `schema` or `data_model` arguments."
            )

        if language_model:
            self.decision_language_model = language_model
            self.action_language_model = language_model
        elif action_language_model and decision_language_model:
            self.decision_language_model = decision_language_model
            self.action_language_model = action_language_model
        else:
            raise ValueError(
                "You must set either `language_model` "
                " or both `action_language_model` and `decision_language_model` arguments."
            )

        self.prompt_template = prompt_template

        if not examples:
            examples = []

        self.examples = examples

        if not instructions:
            instructions = get_default_decision_instructions()

        self.instructions = instructions

        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
    
        if return_inputs_only and return_inputs_with_trajectory:
            raise ValueError(
                "You cannot set both `return_inputs_only` and "
                "`return_inputs_with_trajectory` arguments to true: choose only one."
            )

        self.return_inputs_with_trajectory = return_inputs_with_trajectory
        self.return_inputs_only = return_inputs_only

        assert max_iterations > 0, "The agent must perform at least one decision-making step."
        self.max_iterations = max_iterations

        toolkit = toolkit or []

        self.toolkit = [_ if isinstance(_, Tool) else Tool(_) for _ in toolkit]
        self.labels = [_.name() for _ in self.toolkit]

        if question:
            question = Question(question=question)
        else:
            question = get_default_decision_question(self.toolkit)

        self.question = question

        self.actions = []

        for _ in self.toolkit:
            self.actions.append(
                Action(
                    fn=_._func,
                    language_model=self.action_language_model,
                    prompt_template=self.prompt_template,
                    use_inputs_schema=self.use_inputs_schema,
                    use_outputs_schema=self.use_outputs_schema,
                )
            )
  
        decision_schema = dynamic_enum_on_nested_property(
            ToolDecision.get_schema(),
            "ToolChoice/properties/tool",
            self.labels,
            description="The name of the tool to run from the available toolkit."
        )

        self.decision_maker = Generator(
            schema=decision_schema,
            language_model=self.decision_language_model,
            instructions=self.instructions,
            prompt_template=self.prompt_template,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            name=self.name + "_decision_maker",
        )

        self.response_maker = Generator(
            schema=self.schema,
            language_model=self.action_language_model,
            instructions=["Provide the final response, taking into account all the information gathered."],
            name=f"{self.name}_response_maker",
        )

    async def call(self, inputs, training=False):
        state = inputs

        question = JsonDataModel(
            json=self.question.get_json(),
            data_model=Question,
            name=self.name + "_question"
        )

        state = await ops.concat(
            state,
            question,
            name=self.name + "_inputs_with_question",
        )

        for _ in range(self.max_iterations):                
            decision = await self.decision_maker(state, training=training)
            choices = decision.get("choices", [])

            if not choices:
                break

            futures = []

            for choice in choices:
                tool = choice.get("tool")
                purpose = choice.get("purpose")

                purpose = Purpose(purpose=purpose)
                purpose = JsonDataModel(
                    json=purpose.get_json(),
                    data_model=Purpose,
                )

                action = self.actions[self.labels.index(tool)]

                futures.append(action(purpose, training=training))

            messages = await asyncio.gather(*futures)

            if len(messages) == 1:
                tool_message = messages[0]
            else:
                tool_message = messages[0]

                for i in range(1, len(messages)):
                    tool_message = await ops.concat(
                        tool_message, 
                        messages[i],
                        name=f"{self.name}_tool_message_{i}"
                    )

            state = await ops.concat(state, tool_message)

        response = await self.response_maker(state, training=training)

        if self.return_inputs_with_trajectory:
            response = await ops.concat(state, response)
            # FIXME: response.factorize()

        if self.return_inputs_only:
            response = await ops.concat(inputs, response)

        return response

    async def compute_output_spec(self, inputs, training=False):
        state = inputs

        question = JsonDataModel(
            json=self.question.get_json(),
            data_model=Question,
            name=self.name + "_question"
        )

        state = await ops.concat(
            state,
            question,
            name=self.name + "_inputs_with_question",
        )

        # Simulate the iterative decision-making loop
        for _ in range(self.max_iterations):
            decision_spec = await self.decision_maker.compute_output_spec(state)
            
            # Simulate tool execution if actions exist
            if self.actions:
                action_specs = []

                for action in self.actions:
                    # Create a dummy purpose for computing output spec
                    purpose = Purpose(purpose="dummy purpose for output spec computation")
                    purpose = JsonDataModel(
                        json=purpose.get_json(),
                        data_model=Purpose,
                    )
                    spec = await action.compute_output_spec(purpose)
                    action_specs.append(spec)
                
                if len(action_specs) == 1:
                    combined_spec = action_specs[0]
                else:
                    combined_spec = action_specs[0]
                    for i in range(1, len(action_specs)):
                        combined_spec = await ops.concat(
                            combined_spec, 
                            action_specs[i],
                            name=f"{self.name}_combined_spec_{i}"
                        )
                state = await ops.concat(state, combined_spec)

        response = await self.response_maker.compute_output_spec(state)

        if self.return_inputs_with_trajectory:
            response = await ops.concat(state, response)

        if self.return_inputs_only:
            response = await ops.concat(inputs, response)

        return response

    def get_config(self):
        config = {
            "schema": self.schema,
            "toolkit": self.toolkit,
            "question": self.question,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "return_inputs_with_trajectory": self.return_inputs_with_trajectory,
            "return_inputs_only": self.return_inputs_only,
            "max_iterations": self.max_iterations,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }

        language_model_config = {}
        language_model_config["decision_language_model"] = serialize_synalinks_object(
            self.decision_language_model
        )
        language_model_config["action_language_model"] = serialize_synalinks_object(
            self.action_language_model
        )

        return {**config, **language_model_config}

    @classmethod
    def from_config(cls, config):
        language_model = deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(
            language_model=language_model,
            **config,
        )
