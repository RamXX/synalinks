import asyncio


from synalinks.src import ops
from synalinks.src.modules.module import Module
from synalinks.src.utils.tool_utils import Tool
from synalinks.src.modules.core.action import Action
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.core.multi_decision import MultiDecision

def get_tool_selection_question():
    '''
    Default question prompt for tool selection in the React agent
    Returns:
        str: The question asking the LM to choose tools to execute
    '''
    return "Choose one or more functions to use next in parallel based on their name."

def get_tool_selection_instruction():
    '''
    Behavioral instructions for tool selection decisions.
    Returns:
        list: List of instruction strings for the tool selection process
    '''
    return [
        "Always reflect on your previous actions to know what to do.",
        "You can call the same tool multiple times if needed.",
        "Each call should have a clear reasoning explaining why that specific tool is needed.",
        "If no tools are needed, return an empty calls list.",
    ]

def get_reasoning_instructions(tool_name):
    '''
    Instructions for generating reasoning for a specific tool
    Args:
        tool_name (str): The name of the tool to generate reasoning for
    Returns:
        list: List of instruction strings for reasoning generation
    '''
    return [
        f"Provide reasoning for why you need to use {tool_name}",
        "Be specific about what you expect this tool to accomplish",
        "Consider the current context and previous actions",
    ]

class ParallelReACTAgent(Module):

    '''
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
            instructions = get_tool_selection_instruction()
        self.instructions = instructions

        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        if return_inputs_only and return_inputs_with_trajectory:
            raise ValueError(
                "You cannot set both "
                "`return_inputs_only` and `return_inputs_with_trajectory` "
                "arguments to True. Choose only one."
            )
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
                'No functions selected'
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

        self.tool_selector = MultiDecision(
            question=self.question,
            labels=self.labels,
            language_model=self.decision_language_model,
            instructions=self.instructions,
            name=self.name + "_tool_selector",
        )

        self.reasoning_generators = {}
        for tool_name in self.labels:
            self.reasoning_generators[tool_name] = Generator(
                schema={"type": "string"},
                language_model=self.decision_language_model,
                instructions=get_reasoning_instructions(tool_name),
                name=f"{self.name}_reasoning_for_{tool_name}",
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
            tool_selection = await self.tool_selector(current_step, training=training)
            selected_tools = tool_selection.get("choices", [])

            tool_calls = []
            for tool_name in selected_tools:
                reasoning = await self.reasoning_generators[tool_name](current_step, training=training)
                tool_calls.append((tool_name, reasoning))

            if not tool_calls:
                break

            tasks = []
            for tool_name, reasoning in tool_calls:
                tool_index = self.labels.index(tool_name)
                action = self.actions[tool_index]
                tasks.append(action(current_step, training=training))

            tool_results = await asyncio.gather(*tasks)

            if tool_results:
                combined_results = await ops.concat(*tool_results)
                current_step = await ops.concat(current_step, combined_results)

        if self.schema:
            final_answer = await self.final_generator(current_step, training=training)

            return final_answer
        else:
            return current_step
