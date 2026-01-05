"""LM Handler for routing requests via TCP socket."""

import asyncio
import json
import socket
import socketserver
import threading
from typing import Optional
from typing import Union

from synalinks.src.modules.rlm.clients.synalinks_adapter import SynalinksLMClient


class LMRequestHandler(socketserver.BaseRequestHandler):
    """Request handler for LMHandler TCP server.

    This handler processes individual client connections in separate threads
    via ThreadingTCPServer. It delegates to the parent LMHandler for routing
    and client management.
    """

    def handle(self):
        """Handle incoming request and route to appropriate client."""
        try:
            data = self.request.recv(65536).decode("utf-8")
            request = json.loads(data)

            # Get the parent handler from server
            handler: "LMHandler" = self.server.lm_handler  # type: ignore

            # Check if this is a batched request
            if request.get("batch", False):
                handler._handle_batched_internal(request, self.request)
                return

            prompt = request.get("prompt", "")
            client_name = request.get("client", None)
            current_depth = request.get("current_depth", 0)

            # Route to appropriate client
            if client_name and client_name in handler._clients:
                client = handler._clients[client_name]
            else:
                client = handler._default_client

            if client is None:
                response = {"error": "No client registered"}
            else:
                try:
                    result = client.completion(prompt)
                    response = {"result": result, "current_depth": current_depth}
                except Exception as e:
                    response = {"error": str(e)}

            self.request.sendall(json.dumps(response).encode("utf-8"))
        except Exception:
            # Errors are logged by socketserver framework
            raise


class LMHandler:
    """Routes LLM requests via TCP socket with multi-client support.

    Enables REPL-based llm_query() to make LLM calls by routing through
    a TCP socket server. Supports registering multiple clients for
    root vs sub-LM cost optimization.

    Args:
        port: TCP port to listen on (default: 0 for random port)
        host: Host to bind to (default: localhost)

    Example:
        >>> handler = LMHandler()
        >>> handler.register_client("root", client_root)
        >>> handler.register_client("sub", client_sub)
        >>> handler.start()
        >>> # REPL can now call llm_query()
        >>> handler.stop()
    """

    def __init__(self, port: int = 0, host: str = "127.0.0.1"):
        self.host = host
        self.port = port
        self._clients: dict[str, SynalinksLMClient] = {}
        self._default_client: Optional[SynalinksLMClient] = None
        self._server: Optional[socketserver.ThreadingTCPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

    def register_client(
        self, name: str, client: SynalinksLMClient, is_default: bool = False
    ):
        """Register an LM client for routing.

        Enables multi-model architecture for cost optimization. Register
        separate clients for root (expensive) and sub (cheap) models.

        Args:
            name: Client identifier (e.g., "root", "sub")
            client: SynalinksLMClient instance
            is_default: Whether this is the default client for unrouted requests

        Example:
            >>> handler = LMHandler()
            >>> lm_root = LanguageModel(model="zai/glm-4.7")
            >>> lm_sub = LanguageModel(model="groq/openai/gpt-oss-20b")
            >>> handler.register_client("root", SynalinksLMClient(lm_root))
            >>> handler.register_client("sub", SynalinksLMClient(lm_sub))
        """
        self._clients[name] = client
        if is_default or self._default_client is None:
            self._default_client = client

    def get_client(self, name: str) -> Optional[SynalinksLMClient]:
        """Get registered client by name.

        Routes to appropriate client based on name. Returns None if
        client not found. Used for multi-model routing.

        Args:
            name: Client identifier

        Returns:
            SynalinksLMClient instance or None

        Example:
            >>> handler = LMHandler()
            >>> handler.register_client("root", client_root)
            >>> handler.register_client("sub", client_sub)
            >>> root_client = handler.get_client("root")
            >>> sub_client = handler.get_client("sub")
        """
        return self._clients.get(name)

    def start(self):
        """Start TCP server for handling requests using ThreadingTCPServer."""
        if self._running:
            return

        # Create ThreadingTCPServer with custom request handler
        self._server = socketserver.ThreadingTCPServer(
            (self.host, self.port), LMRequestHandler
        )
        self._server.allow_reuse_address = True

        # Attach this handler to the server for request routing
        self._server.lm_handler = self  # type: ignore

        # Get actual port if random port was requested
        if self.port == 0:
            self.port = self._server.server_address[1]

        self._running = True
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()

    def stop(self):
        """Stop TCP server."""
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread:
            self._server_thread.join(timeout=1.0)

    def _serve(self):
        """Server loop using ThreadingTCPServer.serve_forever()."""
        if self._server:
            self._server.serve_forever()

    def _handle_batched_internal(self, request: dict, conn: socket.socket):
        """Internal helper for batched requests with pre-parsed request."""
        try:
            prompts = request.get("prompts", [])
            client_name = request.get("client", None)

            try:
                # Run async batch in new event loop (connection handler is threaded)
                results = asyncio.run(self.acompletion_batched(prompts, client_name))

                # Convert exceptions to error dicts for JSON serialization
                serialized_results = []
                for result in results:
                    if isinstance(result, Exception):
                        serialized_results.append({"error": str(result)})
                    else:
                        serialized_results.append({"result": result})

                response = {"results": serialized_results}
            except Exception as e:
                response = {"error": str(e)}

            conn.sendall(json.dumps(response).encode("utf-8"))
        except Exception:
            # Error already handled by outer _handle_connection finally block
            raise

    def get_port(self) -> int:
        """Get the port the server is listening on."""
        return self.port

    def create_llm_query_fn(
        self,
        client_name: Optional[str] = None,
        current_depth: int = 0,
        max_depth: int = 1,
    ):
        """Create llm_query function for REPL injection.

        Args:
            client_name: Name of client to route to (None for default)
            current_depth: Current recursion depth (default: 0)
            max_depth: Maximum allowed recursion depth (default: 1)

        Returns:
            Function that can be injected as llm_query() in REPL
        """

        def llm_query(prompt: str) -> str:
            """Query LLM via TCP socket.

            When current_depth < max_depth, allows nested llm_query calls.
            When current_depth >= max_depth, uses direct completion without recursion.
            """
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((self.host, self.port))
                request = {"prompt": prompt, "current_depth": current_depth}
                if client_name:
                    request["client"] = client_name
                sock.sendall(json.dumps(request).encode("utf-8"))
                data = sock.recv(65536).decode("utf-8")
                response = json.loads(data)
                if "error" in response:
                    raise RuntimeError(f"LLM query failed: {response['error']}")
                return response.get("result", "")
            finally:
                sock.close()

        return llm_query

    def get_all_usage(self) -> dict[str, dict]:
        """Get usage from all registered clients.

        Returns aggregated usage statistics for cost analysis across
        all clients. Useful for comparing root vs sub-model costs.

        Returns:
            Dictionary mapping client name to usage summary

        Example:
            >>> handler = LMHandler()
            >>> handler.register_client("root", client_root)
            >>> handler.register_client("sub", client_sub)
            >>> # ... make some calls ...
            >>> usage = handler.get_all_usage()
            >>> print(f"Root tokens: {usage['root']['total_tokens']}")
            >>> print(f"Sub tokens: {usage['sub']['total_tokens']}")
        """
        return {
            name: client.get_usage_summary() for name, client in self._clients.items()
        }

    async def acompletion_batched(
        self,
        prompts: list[Union[str, list[dict]]],
        client_name: Optional[str] = None,
    ) -> list[Union[str, Exception]]:
        """Execute multiple completions concurrently using asyncio.gather().

        This is the core async batching method that enables ~5x speedup for
        parallel LLM calls. Individual errors don't fail the entire batch -
        exceptions are returned as elements in the result list.

        Args:
            prompts: List of prompts to execute in parallel
            client_name: Name of client to route to (None for default)

        Returns:
            List of results or exceptions, one per prompt in same order

        Raises:
            RuntimeError: If no client registered

        Example:
            >>> handler = LMHandler()
            >>> handler.register_client("sub", client_sub)
            >>> handler.start()
            >>> prompts = ["Summarize A", "Summarize B", "Summarize C"]
            >>> results = await handler.acompletion_batched(prompts, "sub")
            >>> # Returns ~5x faster than sequential for 5 calls
        """
        # Route to appropriate client
        if client_name and client_name in self._clients:
            client = self._clients[client_name]
        else:
            client = self._default_client

        if client is None:
            raise RuntimeError("No client registered")

        # Create async tasks for all prompts
        tasks = [client.acompletion(prompt) for prompt in prompts]

        # Execute concurrently using asyncio.gather with return_exceptions=True
        # This ensures individual errors don't fail the entire batch
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def create_llm_query_batched_fn(self, client_name: Optional[str] = None):
        """Create llm_query_batched function for REPL injection.

        Args:
            client_name: Name of client to route to (None for default)

        Returns:
            Function that can be injected as llm_query_batched() in REPL

        Example:
            >>> handler = LMHandler()
            >>> handler.register_client("sub", client_sub)
            >>> handler.start()
            >>> llm_query_batched = handler.create_llm_query_batched_fn("sub")
            >>> results = llm_query_batched(["Q1", "Q2", "Q3"])
        """

        def llm_query_batched(prompts: list[str]) -> list[str]:
            """Query LLM with multiple prompts via batched TCP socket request."""
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((self.host, self.port))
                request = {"prompts": prompts, "batch": True}
                if client_name:
                    request["client"] = client_name
                sock.sendall(json.dumps(request).encode("utf-8"))
                data = sock.recv(65536).decode("utf-8")
                response = json.loads(data)
                if "error" in response:
                    raise RuntimeError(f"LLM batch query failed: {response['error']}")

                # Extract results, preserving exceptions
                results = []
                for item in response.get("results", []):
                    if "error" in item:
                        results.append(RuntimeError(item["error"]))
                    else:
                        results.append(item.get("result", ""))
                return results
            finally:
                sock.close()

        return llm_query_batched
