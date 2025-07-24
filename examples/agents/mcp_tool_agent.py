# import multiprocessing
import asyncio

from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.language_models.language_model import LanguageModel
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.programs import Program
from synalinks.src.modules import Input
from synalinks.src.utils.mcp._test_common import run_streamable_server_multiprocessing
from synalinks.src.utils.mcp.client import MultiServerMCPClient
from mcp.server.fastmcp import FastMCP

# Multirpocessing for MacOS
# multiprocessing.set_start_method('fork')

class Query(DataModel):
    """Input query data model"""
    query: str = Field(
        description="The user query",
    )

class FinalAnswer(DataModel):
    """Final answer data model"""
    answer: str = Field(
        description="The correct final answer",
    )


class MCPMathAgent:
    """MCP-based Math Agent with ReACT capabilities"""
    
    def __init__(self):
        self.status_server = None
        self.math_server = None
        self.status_server_context = None
        self.math_server_context = None
        self.client = None
        self.program = None
        
    def setup_servers(self):
        """Setup mock MathMCP servers with tools"""
        
        # Status server setup
        self.status_server = FastMCP(port=8182)

        @self.status_server.tool()
        def get_status() -> str:
            """Get server status"""
            return "Server is running"

        # Math server setup
        self.math_server = FastMCP(port=8183)
        
        @self.math_server.tool()
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together"""
            return a + b
            
    async def start_servers(self):
        """Start the MCP servers"""
        try:
            # Set up server contexts
            self.status_server_context = run_streamable_server_multiprocessing(self.status_server)
            self.math_server_context = run_streamable_server_multiprocessing(self.math_server)
            
            # Enter contexts
            self.status_server_context.__enter__()
            self.math_server_context.__enter__()

        except Exception as e:
            return {
                "result": None,
                "log": f"Failed to start server: {e}",
            }
        
    async def setup_client(self):
        """Setup MCP client with server connections"""
        
        status_connection = {
            "url": "http://localhost:8182/mcp/",
            "transport": "streamable_http",
        }
        
        math_connection = {
            "url": "http://localhost:8183/mcp/",
            "transport": "streamable_http",
        }
            
        try:
            self.client = MultiServerMCPClient({
                "status": status_connection,
                "math": math_connection,
            })

        except Exception as e:
            return {
                "result": None,
                "log": f"Failed to client: {e}",
            }

    async def run_example_agent(self):
        """Create the autonomous agent with MCP tools"""
        try:
            assert self.client
            tools = await self.client.get_tools()
            
            for tool in tools:
                tool._func.__name__ = tool._func.__name__.replace('/', '_')

            language_model = LanguageModel(
                model="openai/gpt-4o-mini",
            )

            inputs = Input(data_model=Query)
            outputs = await FunctionCallingAgent(
                data_model=FinalAnswer,
                tools=tools,
                language_model=language_model,
                max_iterations=5,
                autonomous=True,
            )(inputs)
            
            self.program = Program(
                inputs=inputs,
                outputs=outputs,
                name="mcp_math_agent",
                description="A math agent that can use an external calculator",
            )

            input_query = Query(query="How much is 152648 + 485 and what is the server status?")
            response = await self.program(input_query)

            print(response.prettify_json())
            
        except Exception as e:
            return {
                "result": None,
                "log": f"Failed to agent: {e}",
            }

async def main():
    mcp_agent = MCPMathAgent()

    mcp_agent.setup_servers()
    await mcp_agent.start_servers()
    await mcp_agent.setup_client()
    await mcp_agent.run_example_agent()

if __name__ == "__main__":
    asyncio.run(main())
