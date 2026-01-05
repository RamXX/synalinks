"""LM Handler for routing requests via TCP socket."""

import json
import socket
import threading
from typing import Optional

from synalinks.src.modules.rlm.clients.synalinks_adapter import SynalinksLMClient


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
        self._server_socket: Optional[socket.socket] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

    def register_client(
        self, name: str, client: SynalinksLMClient, is_default: bool = False
    ):
        """Register an LM client for routing.

        Args:
            name: Client identifier (e.g., "root", "sub")
            client: SynalinksLMClient instance
            is_default: Whether this is the default client for unrouted requests
        """
        self._clients[name] = client
        if is_default or self._default_client is None:
            self._default_client = client

    def start(self):
        """Start TCP server for handling requests."""
        if self._running:
            return

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)

        # Get actual port if random port was requested
        if self.port == 0:
            self.port = self._server_socket.getsockname()[1]

        self._running = True
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()

    def stop(self):
        """Stop TCP server."""
        self._running = False
        if self._server_socket:
            self._server_socket.close()
        if self._server_thread:
            self._server_thread.join(timeout=1.0)

    def _serve(self):
        """Server loop handling incoming connections."""
        while self._running:
            try:
                self._server_socket.settimeout(0.5)
                conn, addr = self._server_socket.accept()
                threading.Thread(
                    target=self._handle_connection, args=(conn,), daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception:
                if self._running:
                    raise

    def _handle_connection(self, conn: socket.socket):
        """Handle a single client connection."""
        try:
            data = conn.recv(65536).decode("utf-8")
            request = json.loads(data)

            prompt = request.get("prompt", "")
            client_name = request.get("client", None)

            # Route to appropriate client
            if client_name and client_name in self._clients:
                client = self._clients[client_name]
            else:
                client = self._default_client

            if client is None:
                response = {"error": "No client registered"}
            else:
                try:
                    result = client.completion(prompt)
                    response = {"result": result}
                except Exception as e:
                    response = {"error": str(e)}

            conn.sendall(json.dumps(response).encode("utf-8"))
        finally:
            conn.close()

    def get_port(self) -> int:
        """Get the port the server is listening on."""
        return self.port

    def create_llm_query_fn(self, client_name: Optional[str] = None):
        """Create llm_query function for REPL injection.

        Args:
            client_name: Name of client to route to (None for default)

        Returns:
            Function that can be injected as llm_query() in REPL
        """

        def llm_query(prompt: str) -> str:
            """Query LLM via TCP socket."""
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((self.host, self.port))
                request = {"prompt": prompt}
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
