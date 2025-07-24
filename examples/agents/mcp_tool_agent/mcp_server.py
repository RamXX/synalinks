from mcp.server.fastmcp import FastMCP
from synalinks.src.utils.mcp._test_common import run_streamable_server_multiprocessing

class McpServer:
    
    def __init__(self):
        self.status_server = None
        self.math_server = None
        self.status_server_context = None
        self.math_server_context = None

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
