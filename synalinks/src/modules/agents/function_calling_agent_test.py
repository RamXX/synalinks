# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import is_chat_messages
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.input_module import Input
from synalinks.src.programs import Program
from synalinks.src.saving.object_registration import register_synalinks_serializable
from synalinks.src.utils.tool_utils import Tool


@register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate, such as
            '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
            parentheses, and spaces.
    """
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


@register_synalinks_serializable()
async def thinking(thinking: str):
    """Think about something.

    Args:
        thinking (str): Your step by step thinking.
    """
    return {
        "thinking": thinking,
    }


@register_synalinks_serializable()
async def get_weather(location: str):
    """Get weather information for a location.
    Args:
        location (str): The location to get weather for.
    """
    # Mock weather data
    weather_data = {
        "New York": {"temp": 22, "condition": "Sunny"},
        "London": {"temp": 15, "condition": "Cloudy"},
        "Tokyo": {"temp": 28, "condition": "Rainy"},
    }

    if location in weather_data:
        return {
            "location": location,
            "temperature": weather_data[location]["temp"],
            "condition": weather_data[location]["condition"],
            "success": True,
        }
    else:
        return {"location": location, "error": "Location not found", "success": False}


@register_synalinks_serializable()
async def failing_tool(should_fail: bool = True):
    """A tool that intentionally fails for testing error handling.
    Args:
        should_fail (bool): Whether the tool should fail.
    """
    if should_fail:
        raise ValueError("This tool was designed to fail")
    return {"status": "success"}


class FunctionCallingAgentTest(testing.TestCase):
    async def test_agent_instantiation(self):
        """Test basic agent instantiation."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
        )(inputs)
        program = Program(
            inputs=inputs,
            outputs=outputs,
            name="function_calling_agent_test",
        )
        self.assertIsNotNone(program)

    @patch("litellm.acompletion")
    async def test_autonomous_mode_simple_calculation(self, mock_completion):
        """Test autonomous mode with a simple calculation."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=True,
            max_iterations=3,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="autonomous_calculation_test",
        )

        tool_calls = {
            "thinking": "Perform simple arithmetic operation by adding the numbers given in the input.",  # noqa: E501
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "152648 + 485",
                }
            ],
        }

        tool_calls_1 = {
            "thinking": "The user has asked for a simple arithmetic operation, specifically adding 152648 and 485. I have already performed the calculation using the 'calculate' tool and obtained the result as 153133.",  # noqa: E501
            "tool_calls": [],
        }

        mock_responses = [
            {"choices": [{"message": {"content": json.dumps(tool_calls)}}]},
            {"choices": [{"message": {"content": json.dumps(tool_calls_1)}}]},
        ]

        mock_completion.side_effect = mock_responses

        input_messages = ChatMessages(
            messages=[
                ChatMessage(
                    role="user",
                    content="How much is 152648 + 485?",
                ),
            ]
        )
        result = await agent(input_messages)

        print("Result:")
        print(result.prettify_json())

        # Verify result structure
        self.assertIsNotNone(result)
        messages = result.get("messages", [])
        self.assertGreater(len(messages), 0)
        self.assertTrue(is_chat_messages(result))

    @patch("litellm.acompletion")
    async def test_autonomous_mode_complex_calculation(self, mock_completion):
        """Test autonomous mode with a more complex calculation."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=True,
            max_iterations=5,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="complex_calculation_test",
        )

        tool_calls = {
            "thinking": "First, I will perform the arithmetic operation as instructed. Let's calculate (150 + 250) * 2 / 4. The order of operations is follow BIDMAS/BODMAS which means Brackets, Orders or Powers, Division and Multiplication, Addition and Subtraction. So, first I will add 150 and 250, then multiply the result by 2, divide it by 4 and finally add 100.",  # noqa: E501
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "(150 + 250) * 2 / 4 + 100",
                }
            ],
        }

        tool_calls_1 = {
            "thinking": "The user provided a mathematical expression to calculate. I performed the operation (150 + 250) * 2 / 4 and then added 100 to the result. Now, the result is 300.",  # noqa: E501
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "(150 + 250) * 2 / 4 + 100",
                }
            ],
        }

        mock_responses = [
            {"choices": [{"message": {"content": json.dumps(tool_calls)}}]},
            {"choices": [{"message": {"content": json.dumps(tool_calls_1)}}]},
        ]

        mock_completion.side_effect = mock_responses

        input_messages = ChatMessages(
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Calculate (150 + 250) * 2 / 4 and then add 100 to the result"
                    ),
                ),
            ]
        )
        result = await agent(input_messages)
        print("Result:")
        print(result.prettify_json())

    # async def test_interactive_mode_single_step(self):
    #     """Test interactive mode with single step execution."""
    #     language_model = LanguageModel(model="ollama/mistral")
    #     tools = [
    #         Tool(calculate),
    #         Tool(thinking),
    #     ]
    #     inputs = Input(data_model=ChatMessages)
    #     outputs = await FunctionCallingAgent(
    #         language_model=language_model,
    #         tools=tools,
    #         autonomous=False,
    #         return_inputs_with_trajectory=True,
    #     )(inputs)
    #     agent = Program(
    #         inputs=inputs,
    #         outputs=outputs,
    #         name="interactive_single_step_test",
    #     )

    #     input_messages = ChatMessages(
    #         messages=[
    #             ChatMessage(
    #                 role="user",
    #                 content="How much is 152648 + 485?",
    #             )
    #         ]
    #     )
    #     result = await agent(input_messages)
    #     print("Interactive Mode Single Step Result:")
    #     print(result.prettify_json())
    #     raise ValueError

    # async def test_interactive_mode_multi_step(self):
    #     """Test interactive mode with multiple steps simulation."""
    #     language_model = LanguageModel(model="ollama/mistral")
    #     tools = [
    #         Tool(calculate),
    #         Tool(thinking),
    #     ]
    #     inputs = Input(data_model=ChatMessages)
    #     outputs = await FunctionCallingAgent(
    #         language_model=language_model,
    #         tools=tools,
    #         autonomous=False,
    #         return_inputs_with_trajectory=True,
    #     )(inputs)
    #     agent = Program(
    #         inputs=inputs,
    #         outputs=outputs,
    #         name="interactive_multi_step_test",
    #     )

    #     input_messages = ChatMessages(
    #         messages=[
    #             ChatMessage(
    #                 role="user",
    #                 content="I need to calculate 100 + 200, then multiply by 3",
    #             )
    #         ]
    #     )

    #     # Simulate multiple interaction steps
    #     max_steps = 3
    #     for step in range(max_steps):
    #         print(f"\n--- Step {step + 1} ---")
    #         result = await agent(input_messages)
    #         print(f"Step {step + 1} Result:")
    #         print(result.prettify_json())

    #         # Get the latest assistant message
    #         messages = result.get("messages", [])
    #         if messages:
    #             last_message = messages[-1]
    #             if last_message.get("role") == "assistant":
    #                 tool_calls = last_message.get("tool_calls", [])
    #                 if not tool_calls:
    #                     print("No more tool calls, stopping interaction")
    #                     break
    #                 # In a real scenario, you would validate/modify tool calls here
    #                 input_messages = result
    #             else:
    #                 break
    #         else:
    #             break

    # async def test_interactive_mode(self):
    #     language_model = LanguageModel(model="ollama/mistral")

    #     tools = [
    #         Tool(calculate),
    #         Tool(thinking),
    #     ]

    #     inputs = Input(data_model=ChatMessages)
    #     outputs = await FunctionCallingAgent(
    #         language_model=language_model,
    #         tools=tools,
    #     )(inputs)

    #     agent = Program(
    #         inputs=inputs,
    #         outputs=outputs,
    #         name="function_calling_agent_test",
    #     )

    #     input_messages = ChatMessages(
    #         messages=[
    #             ChatMessage(
    #                 role="user",
    #                 content="How much is 152648 + 485?",
    #             )
    #         ]
    #     )

    #     result = await agent(input_messages)

    #     print("Result:")
    #     print(result.prettify_json())
