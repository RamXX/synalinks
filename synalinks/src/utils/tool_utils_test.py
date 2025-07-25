# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import saving
from synalinks.src import testing
from synalinks.src.utils.tool_utils import Tool


@saving.object_registration.register_synalinks_serializable()
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


class ToolUtilsTest(testing.TestCase):
    def test_basic_tool(self):
        _ = Tool(calculate)

    async def test_tool_serialization(self):
        tool = Tool(calculate)
        tool_config = tool.get_config()
        new_tool = Tool.from_config(tool_config)

        tool_call = await new_tool("2+2")
        result = tool_call.get("result")

        self.assertTrue(result == 4)
