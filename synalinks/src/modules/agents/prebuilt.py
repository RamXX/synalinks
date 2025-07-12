# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import copy

from typing import List
from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend.config import backend
from synalinks.src.modules.core.action import Action
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.module import Module
from synalinks.src.utils.nlp_utils import to_singular_property
from synalinks.src.utils.tool_utils import Tool

if backend() != "pydantic":
    raise ValueError(
        f"The module requires the Pydantic backend, but current backend is {backend()}. "
    )

from synalinks.src.backend import Field


class ToolChoice(DataModel):
    name: str
    purpose: str = Field(
        description="A clear, specific explanation of what the tool should accomplish."
    )


class ToolDecision(DataModel):
    reasoning: str = Field(
        description="A step-by-step analysis of the current state, what has been done, and what should be done next."
    )
    choices: List[ToolChoice] = Field(
        description="The array of tool calls to run in parallel with their specific purpose."
    )


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


def get_tool_decision_instructions():
    '''
    Behavioral instructions for tool selection decisions.
    Returns:
        list: List of instruction strings for the tool selection process
    '''
    return [
        "Always reflect on your previous actions and their results to avoid redundancy.",
        "You can call the same tool multiple times if needed with different subgoals.",
        "For each tool you select, provide a clear and specific subgoal explaining what you want to achieve.",
        "Be strategic about parallel execution - choose tools that can run simultaneously without dependencies.",
        "If no more tools are needed to complete the task, return an empty tool_choices array.",
        "Consider the context and information already available before selecting tools.",
    ]


@synalinks_export(["synalinks.modules.Agent", "synalinks.Agent"])
class Agent(Module):

    '''
    ReAct agent as a directed acyclic graph that chooses at each step which tools to run.

    Note:
        - Each function must return a JSON object and be asynchrounous.

    Example:
    ```python
    import synalinks
    import asyncio

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    class FinalAnswer(synalinks.DataModel):
        answer: float = synalinks.Field(
            description="The correct final answer",
        )

    async def main():

        async def calculate(expression: str):
            \"""Calculate the result of a mathematical expression.

            Args:
                expression (str): The mathematical expression to calculate, such as
                    '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
                    parentheses, and spaces.
            \"""
            if not all(char in "0123456789+-*/(). " for char in expression):
                return {
                    "result": None,
                    "log": "Error: invalid characters in expression",
                }
            try:
                # Evaluate the mathematical expression safely
                result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
                return {
                    "result": result,
                    "log": "Successfully executed",
                }
            except Exception as e:
                return {
                    "result": None,
                    "log": f"Error: {e}",
                }

        language_model = synalinks.LanguageModel(model="ollama/mistral")

        x0 = synalinks.Input(data_model=Query)
        x1 = await synalinks.ReACTAgent(
            data_model=FinalAnswer,
            language_model=language_model,
            functions=[calculate],
            max_iterations=3,
        )(x0)

        program = synalinks.Program(
            inputs=x0,
            outputs=x1,
            name="math_agent",
            description="A math agent that can use a calculator",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    References:
        - [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

    Args:
        schema (dict): The JSON schema to use for the final answer.
            If not provided, it will use the `output_data_model` argument.
        data_model (DataModel | JsonDataModel | SymbolicDataModel): Optional.
            The data model to use for the final answer.
            If None provided, the Agent will return a ChatMessage-like data model.
        functions (list): A list of Python functions for the agent to choose from.
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
    '''

    def __init__(
        self,
        schema=None,
        data_model=None,
        functions=None,
        question=None,
        language_model=None,
        decision_language_model=None,
        action_language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        return_inputs_with_trajectory=False,
        return_inputs_only=False,
        max_iterations=5,
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

        if language_model:
            self.decision_language_model = language_model
            self.action_language_model = language_model
        elif action_language_model and decision_language_model:
            self.decision_language_model = decision_language_model
            self.action_language_model = action_language_model
        else:
            raise ValueError(
                "You must set either `language_model` "
                " or both `action_language_model` and `decision_language_model`."
            )

        self.prompt_template = prompt_template

        if not examples:
            examples = []
        self.examples = examples

        if not instructions:
            instructions = get_tool_decision_instructions()
        self.instructions = instructions

        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        if return_inputs_only and return_inputs_with_trajectory:
            raise ValueError(
                "You cannot set both "
                "`return_inputs_only` and `return_inputs_with_trajectory` "
                "arguments to True. Choose only one."
            )
        #TODO: Needs to be implemented into the actual agent output
        self.return_inputs_with_trajectory = return_inputs_with_trajectory
        self.return_inputs_only = return_inputs_only

        assert max_iterations > 1
        self.max_iterations = max_iterations

        if not question:
            question = get_tool_selection_question()
        self.question = question

        self.labels = []

        self.functions = functions or []
        if self.functions == []:
            raise ValueError(
                'Agent requires at least one function to operate'
            )
        for fn in self.functions:
            self.labels.append(Tool(fn).name())

        self.actions = []
        for fn in self.functions:
            self.actions.append(
                Action(
                    fn=fn,
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
        self.decision = Generator(
            schema=decision_schema,
            language_model=self.decision_language_model,
            instructions=self.instructions,
            prompt_template=self.prompt_template,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            name=self.name + "_generator",
        )

        self.final_generator = Generator(
            schema=self.schema,
            language_model=self.action_language_model,
            instructions=["Provide the final answer based on all the information gathered."],
            name=f"{self.name}_final_answer",
        )

    async def call(self, inputs, training=False):
        current_step = inputs

        for _ in range(self.max_iterations):
            inputs = await ops.concat(
                inputs,
                {"question": self.question},
                name=self.name + "_inputs_with_question",
            )
            decision_result = await self.decision(current_step, training=training)
            tool_choices = decision_result.get("tool_choices", [])

            if not tool_choices:
                break

            #TODO: Yoan feedback? Is the action implementation ok like that?
            tasks = []
            for tool_choice in tool_choices:
                tool_name = tool_choice.get("tool")
                tool_index = self.labels.index(tool_name)
                action = self.actions[tool_index]
                tasks.append(action(current_step, training=training))

            tool_results = await asyncio.gather(*tasks)

            combined_results = await ops.concat(*tool_results)
            current_step = await ops.concat(current_step, combined_results)

        final_answer = await self.final_generator(current_step, training=training)

        return final_answer

