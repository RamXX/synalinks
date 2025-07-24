import asyncio
import synalinks
from synalinks.src.utils.mcp.client import MultiServerMCPClient

class Query(synalinks.DataModel):
    """Input query data model"""
    query: str = synalinks.Field(
        description="The user query",
    )

class FinalAnswer(synalinks.DataModel):
    """Final answer data model"""
    answer: str = synalinks.Field(
        description="The correct final answer",
    )

client = MultiServerMCPClient({
    "math": {
        "url": "http://127.0.0.1:8183/mcp/",
        "transport": "streamable_http",
    }
})

async def main():
    tools = await client.get_tools()

    for tool in tools:
        tool._func.__name__ = tool._func.__name__.replace('/', '_')

    language_model = synalinks.LanguageModel(
        model="openai/gpt-4o-mini",
    )

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=FinalAnswer,
        tools=tools,
        language_model=language_model,
        max_iterations=5,
        autonomous=True,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="mcp_math_agent",
        description="A math agent that can use an external calculator",
    )

    input_query = Query(query="How much is 152648 + 485 and 34 * 5?")
    response = await program(input_query)

    print(response.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
