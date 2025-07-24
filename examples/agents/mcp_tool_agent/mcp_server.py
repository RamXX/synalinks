from fastmcp import FastMCP

mcp = FastMCP("Example Math MCP Server")

@mcp.tool
async def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@mcp.tool
async def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8183
    )
