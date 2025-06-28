# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import platform
import unittest
from unittest.mock import AsyncMock, MagicMock

import httpx
from mcp.server import FastMCP
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
    Tool as MCPTool,
    ToolAnnotations,
)

from synalinks.src import testing
from synalinks.src.utils.mcp._test_common import run_streamable_server_multiprocessing
from synalinks.src.utils.mcp.client import MultiServerMCPClient
from synalinks.src.utils.mcp.tools import (
    _convert_call_tool_result,
    convert_mcp_tool_to_synalinks_tool,
    load_mcp_tools,
    ToolException,
)
from synalinks.src.utils.tool_utils import Tool


class MCPToolsTest(testing.TestCase):
    def test_convert_empty_text_content(self):
        result = CallToolResult(
            content=[],
            isError=False,
        )

        tool_message = _convert_call_tool_result(result)

        self.assertEqual(tool_message["response"], "")

    def test_convert_single_text_content(self):
        result = CallToolResult(
            content=[TextContent(type="text", text="test result")],
            isError=False,
        )

        tool_message = _convert_call_tool_result(result)

        self.assertEqual(tool_message["response"], "test result")

    def test_convert_multiple_text_contents(self):
        result = CallToolResult(
            content=[
                TextContent(type="text", text="result 1"),
                TextContent(type="text", text="result 2"),
            ],
            isError=False,
        )

        tool_message = _convert_call_tool_result(result)

        self.assertEqual(tool_message["response"], ["result 1", "result 2"])

    def test_convert_with_non_text_content(self):
        image_content = ImageContent(type="image", mimeType="image/png", data="base64data")
        resource_content = EmbeddedResource(
            type="resource",
            resource=TextResourceContents(uri="resource://test", mimeType="text/plain", text="hi"),
        )

        result = CallToolResult(
            content=[
                TextContent(type="text", text="text result"),
                image_content,
                resource_content,
            ],
            isError=False,
        )

        tool_message = _convert_call_tool_result(result)

        self.assertEqual(tool_message["response"], "text result")
        self.assertNotIn("artifact", tool_message)

    def test_convert_with_error(self):
        result = CallToolResult(
            content=[TextContent(type="text", text="error message")],
            isError=True,
        )

        with self.assertRaises(ToolException) as exc_info:
            _convert_call_tool_result(result)

        self.assertEqual(str(exc_info.exception), "error message")

    async def test_convert_mcp_tool_to_synalinks_tool(self):
        tool_input_schema = {
            "properties": {
                "param1": {"title": "Param1", "type": "string"},
                "param2": {"title": "Param2", "type": "integer"},
            },
            "required": ["param1", "param2"],
            "title": "ToolSchema",
            "type": "object",
        }

        session = AsyncMock()
        session.call_tool.return_value = CallToolResult(
            content=[TextContent(type="text", text="tool result")],
            isError=False,
        )

        mcp_tool = MCPTool(
            name="test_tool",
            description="Test tool description",
            inputSchema=tool_input_schema,
        )

        synalinks_tool = convert_mcp_tool_to_synalinks_tool(session, mcp_tool)

        self.assertIsInstance(synalinks_tool, Tool)
        self.assertEqual(synalinks_tool.name(), "test_tool")
        self.assertStartsWith(synalinks_tool.description(), "Test tool description")

        result = await synalinks_tool.async__call__(param1="test", param2=42)

        session.call_tool.assert_called_once_with("test_tool", {"param1": "test", "param2": 42})

        self.assertEqual(result["response"], "tool result")

    async def test_load_mcp_tools(self):
        tool_input_schema = {
            "properties": {
                "param1": {"title": "Param1", "type": "string"},
                "param2": {"title": "Param2", "type": "integer"},
            },
            "required": ["param1", "param2"],
            "title": "ToolSchema",
            "type": "object",
        }

        session = AsyncMock()
        mcp_tools = [
            MCPTool(
                name="tool1",
                description="Tool 1 description",
                inputSchema=tool_input_schema,
            ),
            MCPTool(
                name="tool2",
                description="Tool 2 description",
                inputSchema=tool_input_schema,
            ),
        ]
        session.list_tools.return_value = MagicMock(tools=mcp_tools, nextCursor=None)

        async def mock_call_tool(tool_name, arguments):
            if tool_name == "tool1":
                return CallToolResult(
                    content=[TextContent(type="text", text=f"tool1 result with {arguments}")],
                    isError=False,
                )
            elif tool_name == "tool2":
                return CallToolResult(
                    content=[TextContent(type="text", text=f"tool2 result with {arguments}")],
                    isError=False,
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        session.call_tool.side_effect = mock_call_tool

        toolkit = await load_mcp_tools(session)

        self.assertEqual(len(toolkit), 2)
        self.assertTrue(all(isinstance(tool, Tool) for tool in toolkit))
        self.assertEqual(toolkit[0].name(), "tool1")
        self.assertStartsWith(toolkit[0].description(), "Tool 1 description")
        self.assertEqual(toolkit[1].name(), "tool2")
        self.assertStartsWith(toolkit[1].description(), "Tool 2 description")

        result1 = await toolkit[0].async__call__(param1="test1", param2=1)
        self.assertEqual(result1["response"], "tool1 result with {'param1': 'test1', 'param2': 1}")

        result2 = await toolkit[1].async__call__(param1="test2", param2=2)
        self.assertEqual(result2["response"], "tool2 result with {'param1': 'test2', 'param2': 2}")

    async def test_load_mcp_tools_with_namespace(self):
        tool_input_schema = {
            "properties": {
                "param1": {"title": "Param1", "type": "string"},
                "param2": {"title": "Param2", "type": "integer"},
            },
            "required": ["param1", "param2"],
            "title": "ToolSchema",
            "type": "object",
        }

        namespace = "testing"

        session = AsyncMock()
        mcp_tools = [
            MCPTool(
                name="tool1",
                description="Tool 1 description",
                inputSchema=tool_input_schema,
            ),
            MCPTool(
                name="tool2",
                description="Tool 2 description",
                inputSchema=tool_input_schema,
            ),
        ]
        session.list_tools.return_value = MagicMock(tools=mcp_tools, nextCursor=None)

        async def mock_call_tool(tool_name, arguments):
            if tool_name == "tool1":
                return CallToolResult(
                    content=[TextContent(type="text", text=f"tool1 result with {arguments}")],
                    isError=False,
                )
            elif tool_name == "tool2":
                return CallToolResult(
                    content=[TextContent(type="text", text=f"tool2 result with {arguments}")],
                    isError=False,
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        session.call_tool.side_effect = mock_call_tool

        toolkit = await load_mcp_tools(session, namespace=namespace)

        self.assertEqual(len(toolkit), 2)
        self.assertTrue(all(isinstance(tool, Tool) for tool in toolkit))
        self.assertEqual(toolkit[0].name(), f"{namespace}/tool1")
        self.assertStartsWith(toolkit[0].description(), "Tool 1 description")
        self.assertEqual(toolkit[1].name(), f"{namespace}/tool2")
        self.assertStartsWith(toolkit[1].description(), "Tool 2 description")

        result1 = await toolkit[0].async__call__(param1="test1", param2=1)
        self.assertEqual(result1["response"], "tool1 result with {'param1': 'test1', 'param2': 1}")

        result2 = await toolkit[1].async__call__(param1="test2", param2=2)
        self.assertEqual(result2["response"], "tool2 result with {'param1': 'test2', 'param2': 2}")

    @unittest.skipUnless(platform.system() == "Linux", "server tests require Linux (multiprocessing/pickling issues on Windows)")
    async def test_load_mcp_tools_with_annotations(self):
        server = FastMCP(port=8181)

        @server.tool(
            annotations=ToolAnnotations(title="Get Time", readOnlyHint=True, idempotentHint=False)
        )
        def get_time() -> str:
            """Get current time"""
            return "5:20:00 PM EST"

        with run_streamable_server_multiprocessing(server):
            client = MultiServerMCPClient(
                {
                    "time": {
                        "url": "http://localhost:8181/mcp/",
                        "transport": "streamable_http",
                    },
                }
            )
            tools = await client.get_tools(server_name="time")
            self.assertEqual(len(tools), 1)
            tool = tools[0]
            self.assertEqual(tool.name(), "get_time")

    @unittest.skipUnless(platform.system() == "Linux", "server tests require Linux (multiprocessing/pickling issues on Windows)")
    async def test_load_mcp_tools_with_custom_httpx_client_factory(self):
        server = FastMCP(port=8182)

        @server.tool()
        def get_status() -> str:
            """Get server status"""
            return "Server is running"

        def custom_httpx_client_factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            """Custom factory for creating httpx.AsyncClient with specific configuration."""
            return httpx.AsyncClient(
                headers=headers,
                timeout=timeout or httpx.Timeout(30.0),
                auth=auth,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )

        with run_streamable_server_multiprocessing(server):
            client = MultiServerMCPClient(
                {
                    "status": {
                        "url": "http://localhost:8182/mcp/",
                        "transport": "streamable_http",
                        "httpx_client_factory": custom_httpx_client_factory,
                    },
                }
            )

            tools = await client.get_tools(server_name="status")
            self.assertEqual(len(tools), 1)
            tool = tools[0]
            self.assertEqual(tool.name(), "get_status")

            result = await tool.async__call__()
            self.assertEqual(result["response"], "Server is running")
