from typing import Dict, Any
from fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")


@mcp.tool
async def calculate(expression: str) -> Dict[str, Any]:
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate, such as
            '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
            parentheses, and spaces.
    """
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {
            "result": None,
            "log": (
                "Error: invalid characters in expression. "
                "The expression can only contain numbers, operators (+, -, *, /),"
                " parentheses, and spaces NOT letters."
            ),
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


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8183,
    )
