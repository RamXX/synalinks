import synalinks
import asyncio
import litellm

litellm._turn_on_debug()

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

async def main():

    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    mcp_client = synalinks.MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["./examples/agents/mcp_tool_agent/mcp_server.py"],
                "transport": "stdio",
            },
        }
    )
    
    tools = await mcp_client.get_tools()
    import json
    print(json.dumps(tools[0].get_tool_schema(), indent=2))
    
#     inputs = synalinks.Input(data_model=Query)
#     outputs = await synalinks.FunctionCallingAgent(
#         data_model=FinalAnswer,
#         tools=tools,
#         language_model=language_model,
#         max_iterations=5,
#         autonomous=True,
#     )(inputs)

#     agent = synalinks.Program(
#         inputs=inputs,
#         outputs=outputs,
#         name="mcp_math_agent",
#         description="A math agent that can use an external calculator",
#     )

#     input_query = Query(query="How much is 152648 + 485 and what is the server status?")
#     response = await agent(input_query)

#     print(response.prettify_json())

if __name__ == "__main__":
    asyncio.run(main())