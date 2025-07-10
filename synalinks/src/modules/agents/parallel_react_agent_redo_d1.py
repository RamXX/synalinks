import asyncio
import copy
import json

from typing import List
from synalinks.src import ops
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_enum
from synalinks.src.modules.module import Module
from synalinks.src.utils.tool_utils import Tool
from synalinks.src.modules.core.action import Action
from synalinks.src.modules.core.generator import Generator
from synalinks.src.backend import DataModel, Field

class ToolChoice(DataModel):
    tool: str = Field(
        description="Name of the specific tool to execute from the available functions."
    )
    subgoal: str = Field(
        description="Clear, specific explanation of what you want to achieve with this tool call and why it's needed at this step."
    )

class MultiDecisionAnswer(DataModel):
    thinking: str = Field(
        description="Step-by-step analysis of the current situation, what has been accomplished, and what needs to be done next."
    )
    tool_choices: List[ToolChoice] = Field(
        description="Array of tools to execute in parallel, each with its specific purpose. Return empty array if no tools are needed."
    )

# TODO: Crate test case for this function. Reference: dynamic_enum()
# def dynamic_enum_nested(schema, property_path, labels, parent_schema=None, description=None):
#     """Update a schema with dynamic Enum at a nested path.

#     Args:
#         schema (dict): The schema to update.
#         property_path (str): Nested path like "tool_choices/items/properties/tool"
#         labels (list): The list of labels (strings).
#         parent_schema (dict, optional): An optional parent schema to use as the base.
#         description (str, optional): An optional description for the enum.

#     Returns:
#         dict: The updated schema with the enum applied to the nested property.
#     """
#     schema = copy.deepcopy(schema)
    
#     if schema.get("$defs"):
#         schema = {"$defs": schema.pop("$defs"), **schema}
#     else:
#         schema = {"$defs": {}, **schema}
    
#     if parent_schema:
#         parent_schema = copy.deepcopy(parent_schema)
    
#     final_prop = property_path.split("/")[-1]
#     title = final_prop.title().replace("_", " ")
    
#     if description:
#         enum_definition = {
#             "enum": labels,
#             "description": description,
#             "title": title,
#             "type": "string",
#         }
#     else:
#         enum_definition = {
#             "enum": labels,
#             "title": title,
#             "type": "string",
#         }
    
#     if parent_schema:
#         parent_schema["$defs"].update({title: enum_definition})
#     else:
#         schema["$defs"].update({title: enum_definition})
    
#     path_parts = property_path.split("/")
#     current = schema
    
#     for part in path_parts[:-1]:
#         if part == "items":
#             current = current.setdefault("items", {})
#         elif part == "properties":
#             current = current.setdefault("properties", {})
#         else:
#             current = current.setdefault(part, {})

#     current[final_prop] = {"$ref": f"#/$defs/{title}"}
        
#     return parent_schema if parent_schema else schema

def dynamic_enum_nested(schema, property_path, labels, parent_schema=None, description=None):
    """Update a schema with dynamic Enum at a nested path.
 
    Args:
        schema (dict): The schema to update.
        property_path (str): Nested path like "tool_choices/items/properties/tool"
        labels (list): The list of labels (strings).
        parent_schema (dict, optional): An optional parent schema to use as the base.
        description (str, optional): An optional description for the enum.

    Returns:
        dict: The updated schema with the enum applied to the nested property.
    """
    schema = copy.deepcopy(schema)
    
    # Ensure $defs is at the top level
    if schema.get("$defs"):
        schema = {"$defs": schema.pop("$defs"), **schema}
    else:
        schema = {"$defs": {}, **schema}
    
    if parent_schema:
        parent_schema = copy.deepcopy(parent_schema)
        if not parent_schema.get("$defs"):
            parent_schema["$defs"] = {}
    
    # Create enum definition
    final_prop = property_path.split("/")[-1]
    title = final_prop.title().replace("_", " ")
    
    enum_definition = {
        "enum": labels,
        "title": title,
        "type": "string",
    }
    
    if description:
        enum_definition["description"] = description
    
    # Add enum definition to $defs
    target_schema = parent_schema if parent_schema else schema
    target_schema["$defs"][title] = enum_definition
    
    # Navigate to the nested property and update it
    path_parts = property_path.split("/")
    current = schema
    
    # Navigate through the path, creating missing structure if needed
    for i, part in enumerate(path_parts[:-1]):
        if part == "items":
            # For array items, we need to ensure the items object exists
            if "items" not in current:
                current["items"] = {}
            current = current["items"]
        elif part == "properties":
            # For object properties, ensure properties object exists
            if "properties" not in current:
                current["properties"] = {}
            current = current["properties"]
        else:
            # For regular property navigation
            if part not in current:
                current[part] = {}
            current = current[part]
    
    # Set the final property to reference the enum
    final_prop = path_parts[-1]
    current[final_prop] = {"$ref": f"#/$defs/{title}"}
    
    return parent_schema if parent_schema else schema


def get_tool_selection_question():
    '''
    Default question prompt for tool selection in the React agent
    Returns:
        str: The question asking the LM to choose tools to execute
    '''
    return "Analyze the current situation and decide which tools to use next. Provide your step-by-step thinking and select the appropriate tools with their specific subgoals."

def get_tool_selection_instruction():
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
        print("Initializing ParallelReACTAgent...")
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
                'ParallelReACTAgent requires at least one function to operate'
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
  
        decision_schema = dynamic_enum_nested(
            MultiDecisionAnswer.get_schema(),
            "$defs/tool_choices/items/properties/tool",
            self.labels,
            description="Available tools to choose from"
        )


        # Debug: print the decision schema when the agent is instantiated
        import json
        print("Decision schema:")
        print(json.dumps(decision_schema, indent=2))


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
    def print_decision_schema():
        """Print the decision schema as JSON output"""
    
        # Sample tool labels (replace with your actual tool labels)
        sample_labels = [
            "web_search",
            "file_reader", 
            "data_processor",
            "email_sender",
            "database_query"
        ]
        
        # Generate the decision schema
        decision_schema = dynamic_enum_nested(
            MultiDecisionAnswer.get_schema(),
            "$defs/ToolChoice/properties/tool",
            sample_labels,
            description="Available tools to choose from"
        )
        
        # Print the schema as formatted JSON
        print("Decision Schema JSON Output:")
        print(json.dumps(decision_schema, indent=2))
        return decision_schema
    

    if __name__ == "__main__":
        schema = print_decision_schema()
