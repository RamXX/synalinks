# License Apache 2.0: (c) 2025 Synalinks Team

"""Deno/Pyodide interpreter for secure Python code execution.

This module provides DenoInterpreter, which runs Python code in a sandboxed
WASM environment using Deno and Pyodide. It implements the CodeInterpreter
protocol for use with RLM (Recursive Language Model).

Features:
- Process isolation via Deno subprocess
- WASM sandbox via Pyodide
- JSON-RPC 2.0 protocol for communication
- SUBMIT mechanism for final output
- Thread-safety enforcement
"""

import asyncio
import functools
import inspect
import json
import keyword
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from synalinks.src.api_export import synalinks_export
from synalinks.src.interpreters.base import CodeInterpreter, CodeInterpreterError

__all__ = ["DenoInterpreter"]

logger = logging.getLogger(__name__)

# Simple types that can be serialized to Python literals
SIMPLE_TYPES = (str, int, float, bool, type(None))

# JSON-RPC 2.0 error codes
JSONRPC_PROTOCOL_ERRORS = {
    "ParseError": -32700,
    "InvalidRequest": -32600,
    "MethodNotFound": -32601,
}

JSONRPC_APP_ERRORS = {
    "SyntaxError": -32000,
    "NameError": -32001,
    "TypeError": -32002,
    "ValueError": -32003,
    "AttributeError": -32004,
    "IndexError": -32005,
    "KeyError": -32006,
    "RuntimeError": -32007,
    "CodeInterpreterError": -32008,
    "Unknown": -32099,
}


def _jsonrpc_request(method: str, params: dict, id: Union[int, str]) -> str:
    """Create a JSON-RPC 2.0 request."""
    return json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": id})


def _jsonrpc_notification(method: str, params: Optional[dict] = None) -> str:
    """Create a JSON-RPC 2.0 notification."""
    msg = {"jsonrpc": "2.0", "method": method}
    if params:
        msg["params"] = params
    return json.dumps(msg)


def _jsonrpc_result(result: Any, id: Union[int, str]) -> str:
    """Create a JSON-RPC 2.0 success response."""
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": id})


def _jsonrpc_error(code: int, message: str, id: Union[int, str], data: Optional[dict] = None) -> str:
    """Create a JSON-RPC 2.0 error response."""
    err = {"code": code, "message": message}
    if data:
        err["data"] = data
    return json.dumps({"jsonrpc": "2.0", "error": err, "id": id})


@synalinks_export(["synalinks.interpreters.DenoInterpreter", "synalinks.DenoInterpreter"])
class DenoInterpreter(CodeInterpreter):
    """Secure Python interpreter using Deno and Pyodide (WASM sandbox).

    Implements the CodeInterpreter protocol for secure code execution in a
    WASM-based sandbox. Code runs in an isolated Pyodide environment with
    no access to the host filesystem, network, or environment by default.

    Prerequisites:
        Deno must be installed: https://docs.deno.com/runtime/getting_started/installation/

    Example:

    ```python
    async with DenoInterpreter() as interp:
        result = await interp.execute("print(1 + 2)")  # Returns "3"

    # With tools
    async def my_tool(question: str) -> dict:
        return {"answer": "42"}

    async with DenoInterpreter() as interp:
        result = await interp.execute(
            "print(my_tool(question='test'))",
            tools={"my_tool": my_tool}
        )
    ```

    Args:
        deno_command: Custom command list to launch Deno.
        enable_read_paths: Files/directories to allow reading.
        enable_write_paths: Files/directories to allow writing.
        enable_env_vars: Environment variable names to expose.
        enable_network_access: Domains/IPs to allow network access.
        sync_files: If True, sync sandbox changes back to host files.
        max_output_chars: Maximum output characters before truncation.
    """

    def __init__(
        self,
        deno_command: Optional[List[str]] = None,
        enable_read_paths: Optional[List[str]] = None,
        enable_write_paths: Optional[List[str]] = None,
        enable_env_vars: Optional[List[str]] = None,
        enable_network_access: Optional[List[str]] = None,
        sync_files: bool = True,
        max_output_chars: int = 100_000,
    ) -> None:
        self.enable_read_paths = enable_read_paths or []
        self.enable_write_paths = enable_write_paths or []
        self.enable_env_vars = enable_env_vars or []
        self.enable_network_access = enable_network_access or []
        self.sync_files = sync_files
        self.max_output_chars = max_output_chars

        self._deno_process: Optional[asyncio.subprocess.Process] = None
        self._mounted_files = False
        self._tools_registered = False
        self._registered_tools: Dict[str, Callable] = {}
        self._output_fields: Optional[List[dict]] = None
        self._request_id = 0
        self._env_arg = ""
        self._owner_thread: Optional[int] = None

        if deno_command:
            self.deno_command = list(deno_command)
        else:
            self.deno_command = self._build_deno_command()

    def _get_runner_path(self) -> str:
        """Get path to the Deno runner script.

        Returns just the filename since we run Deno from the interpreters directory.
        """
        return "deno_runner.js"

    def _check_thread_ownership(self) -> None:
        """Ensure this interpreter is only used from a single thread.

        DenoInterpreter is not thread-safe and cannot be shared across threads.
        This method enforces single-thread ownership.

        Raises:
            RuntimeError: If called from a different thread than the owner.
        """
        current_thread = threading.current_thread().ident
        if self._owner_thread is None:
            self._owner_thread = current_thread
        elif self._owner_thread != current_thread:
            raise RuntimeError(
                "DenoInterpreter is not thread-safe and cannot be shared across threads. "
                "Create a separate interpreter instance for each thread."
            )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _get_deno_dir() -> Optional[str]:
        """Get Deno cache directory."""
        if "DENO_DIR" in os.environ:
            return os.environ["DENO_DIR"]

        try:
            result = subprocess.run(
                ["deno", "info", "--json"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return info.get("denoDir")
        except Exception:
            logger.warning("Unable to find Deno cache directory")

        return None

    def _build_deno_command(self) -> List[str]:
        """Build Deno command with appropriate permissions."""
        args = ["deno", "run"]

        # Build read paths - always include current directory for runner.js
        allowed_read_paths = ["."]
        deno_dir = self._get_deno_dir()
        if deno_dir:
            allowed_read_paths.append(deno_dir)

        if self.enable_read_paths:
            allowed_read_paths.extend(self.enable_read_paths)
        if self.enable_write_paths:
            # Need to read files we want to write to as well
            allowed_read_paths.extend(self.enable_write_paths)

        args.append(f"--allow-read={','.join(allowed_read_paths)}")

        if self.enable_env_vars:
            args.append("--allow-env=" + ",".join(self.enable_env_vars))
            self._env_arg = ",".join(self.enable_env_vars)

        if self.enable_network_access:
            args.append(f"--allow-net={','.join(self.enable_network_access)}")

        if self.enable_write_paths:
            args.append(f"--allow-write={','.join(self.enable_write_paths)}")

        args.append(self._get_runner_path())

        if self._env_arg:
            args.append(self._env_arg)

        return args

    async def _ensure_process(self) -> None:
        """Ensure Deno process is running."""
        if self._deno_process is None or self._deno_process.returncode is not None:
            try:
                # Run from the directory containing deno.json
                interpreter_dir = Path(__file__).parent
                self._deno_process = await asyncio.create_subprocess_exec(
                    *self.deno_command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=os.environ.copy(),
                    cwd=str(interpreter_dir),
                )
            except FileNotFoundError as e:
                raise CodeInterpreterError(
                    "Deno not found. Install from: https://docs.deno.com/runtime/getting_started/installation/\n"
                    "  curl -fsSL https://deno.land/install.sh | sh\n"
                    "  # or: brew install deno"
                ) from e

            await self._health_check()

    async def _health_check(self) -> None:
        """Verify the subprocess is alive."""
        self._request_id += 1
        request_id = self._request_id
        ping_msg = _jsonrpc_request("execute", {"code": "print(1+1)"}, request_id)

        self._deno_process.stdin.write((ping_msg + "\n").encode())
        await self._deno_process.stdin.drain()

        response_line = await self._deno_process.stdout.readline()
        if not response_line:
            raise CodeInterpreterError("Deno process not responding")

        response = json.loads(response_line.decode().strip())
        if response.get("result", {}).get("output", "").strip() != "2":
            raise CodeInterpreterError(f"Unexpected health check response: {response_line}")

    async def _mount_files(self) -> None:
        """Mount files into the sandbox."""
        if self._mounted_files:
            return

        paths_to_mount = self.enable_read_paths + self.enable_write_paths
        if not paths_to_mount:
            return

        for path in paths_to_mount:
            if not path:
                continue
            if not os.path.exists(path):
                if path in self.enable_write_paths:
                    Path(path).touch()
                else:
                    raise FileNotFoundError(f"Cannot mount non-existent file: {path}")

            virtual_path = f"/sandbox/{os.path.basename(path)}"
            self._request_id += 1
            request_id = self._request_id

            mount_msg = _jsonrpc_request(
                "mount_file",
                {"host_path": str(path), "virtual_path": virtual_path},
                request_id
            )
            self._deno_process.stdin.write((mount_msg + "\n").encode())
            await self._deno_process.stdin.drain()

            response_line = await self._deno_process.stdout.readline()
            if not response_line:
                raise CodeInterpreterError(f"No response when mounting: {path}")

            response = json.loads(response_line.decode().strip())
            if response.get("id") != request_id:
                raise CodeInterpreterError(f"Response ID mismatch for mount")
            if "error" in response:
                raise CodeInterpreterError(f"Mount failed for {path}: {response['error'].get('message')}")

        self._mounted_files = True

    async def _sync_files(self) -> None:
        """Sync sandbox changes back to host files."""
        if not self.enable_write_paths or not self.sync_files:
            return

        for path in self.enable_write_paths:
            virtual_path = f"/sandbox/{os.path.basename(path)}"
            sync_msg = _jsonrpc_notification(
                "sync_file",
                {"virtual_path": virtual_path, "host_path": str(path)}
            )
            self._deno_process.stdin.write((sync_msg + "\n").encode())
            await self._deno_process.stdin.drain()

    def _extract_parameters(self, fn: Callable) -> List[dict]:
        """Extract parameter info from a callable."""
        sig = inspect.signature(fn)
        params = []
        for name, param in sig.parameters.items():
            p = {"name": name}
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in SIMPLE_TYPES:
                    p["type"] = param.annotation.__name__
            if param.default != inspect.Parameter.empty:
                p["default"] = param.default
            params.append(p)
        return params

    async def _register_tools(
        self,
        tools: Optional[Dict[str, Callable]] = None,
        output_fields: Optional[List[dict]] = None
    ) -> None:
        """Register tools and output fields with the sandbox."""
        # Only re-register if tools/outputs changed
        tools = tools or {}
        needs_register = (
            not self._tools_registered or
            set(tools.keys()) != set(self._registered_tools.keys()) or
            output_fields != self._output_fields
        )

        if not needs_register:
            return

        self._registered_tools = dict(tools)
        self._output_fields = output_fields

        params = {}

        if tools:
            tools_info = []
            for name, fn in tools.items():
                tools_info.append({
                    "name": name,
                    "parameters": self._extract_parameters(fn)
                })
            params["tools"] = tools_info

        if output_fields:
            params["outputs"] = output_fields

        if not params:
            self._tools_registered = True
            return

        self._request_id += 1
        request_id = self._request_id
        register_msg = _jsonrpc_request("register", params, request_id)

        self._deno_process.stdin.write((register_msg + "\n").encode())
        await self._deno_process.stdin.drain()

        response_line = await self._deno_process.stdout.readline()
        if not response_line:
            raise CodeInterpreterError("No response when registering tools")

        response = json.loads(response_line.decode().strip())
        if "result" not in response or response.get("id") != request_id:
            raise CodeInterpreterError(f"Unexpected register response: {response_line}")

        self._tools_registered = True

    async def _handle_tool_call(self, request: dict) -> None:
        """Handle a tool call request from the sandbox."""
        request_id = request["id"]
        params = request.get("params", {})
        tool_name = params.get("name")
        args = params.get("args", [])
        kwargs = params.get("kwargs", {})

        try:
            if tool_name not in self._registered_tools:
                raise CodeInterpreterError(f"Unknown tool: {tool_name}")

            fn = self._registered_tools[tool_name]
            if asyncio.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)

            is_json = isinstance(result, (list, dict))
            response = _jsonrpc_result(
                {
                    "value": json.dumps(result) if is_json else str(result or ""),
                    "type": "json" if is_json else "string"
                },
                request_id
            )
        except Exception as e:
            error_type = type(e).__name__
            error_code = JSONRPC_APP_ERRORS.get(error_type, JSONRPC_APP_ERRORS["Unknown"])
            response = _jsonrpc_error(error_code, str(e), request_id, {"type": error_type})

        self._deno_process.stdin.write((response + "\n").encode())
        await self._deno_process.stdin.drain()

    def _serialize_value(self, value: Any) -> str:
        """Serialize a Python value for injection."""
        if value is None:
            return "None"
        elif isinstance(value, str):
            return repr(value)
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, (list, dict)):
            return json.dumps(value)
        elif isinstance(value, tuple):
            return json.dumps(list(value))
        elif isinstance(value, set):
            try:
                return json.dumps(sorted(value))
            except TypeError:
                return json.dumps(list(value))
        else:
            raise CodeInterpreterError(f"Unsupported value type: {type(value).__name__}")

    def _inject_variables(self, code: str, variables: Dict[str, Any]) -> str:
        """Insert variable assignments at the top of the code."""
        for key in variables:
            if not key.isidentifier() or keyword.iskeyword(key):
                raise CodeInterpreterError(f"Invalid variable name: '{key}'")

        assignments = [f"{k} = {self._serialize_value(v)}" for k, v in variables.items()]
        return "\n".join(assignments) + "\n" + code if assignments else code

    async def execute(
        self,
        code: str,
        variables: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, Any]:
        """Execute Python code in the sandbox.

        Args:
            code: Python code to execute.
            variables: Variables to inject into the namespace.
            tools: Tool functions callable from sandbox code.

        Returns:
            Dict with keys:
                - success (bool): Whether execution succeeded
                - stdout (str): Captured stdout
                - stderr (str): Captured stderr (always empty for Deno)
                - error (Optional[str]): Error message if any
                - submitted (Optional[dict]): Values from SUBMIT() call
        """
        self._check_thread_ownership()
        variables = variables or {}
        tools = tools or {}

        code = self._inject_variables(code, variables)

        await self._ensure_process()
        await self._mount_files()
        await self._register_tools(tools)

        self._request_id += 1
        execute_request_id = self._request_id
        input_data = _jsonrpc_request("execute", {"code": code}, execute_request_id)

        try:
            self._deno_process.stdin.write((input_data + "\n").encode())
            await self._deno_process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # Process died, restart
            self._tools_registered = False
            self._mounted_files = False
            await self._ensure_process()
            await self._mount_files()
            await self._register_tools(tools)
            self._deno_process.stdin.write((input_data + "\n").encode())
            await self._deno_process.stdin.drain()

        # Read messages until we get final output
        while True:
            output_line = await self._deno_process.stdout.readline()
            if not output_line:
                stderr = await self._deno_process.stderr.read()
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": stderr.decode(),
                    "error": "No output from Deno process",
                    "submitted": None,
                }

            try:
                msg = json.loads(output_line.decode().strip())
            except json.JSONDecodeError:
                await self._sync_files()
                return {
                    "success": True,
                    "stdout": output_line.decode().strip(),
                    "stderr": "",
                    "error": None,
                    "submitted": None,
                }

            # Handle tool calls
            if "method" in msg:
                if msg["method"] == "tool_call":
                    await self._handle_tool_call(msg)
                    continue

            # Handle success
            if "result" in msg:
                if msg.get("id") != execute_request_id:
                    return {
                        "success": False,
                        "stdout": "",
                        "stderr": "",
                        "error": f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}",
                        "submitted": None,
                    }
                result = msg["result"]
                await self._sync_files()

                # Check for SUBMIT
                if "final" in result:
                    return {
                        "success": True,
                        "stdout": "",
                        "stderr": "",
                        "error": None,
                        "submitted": result["final"],
                    }

                output = result.get("output", "")
                if len(output) > self.max_output_chars:
                    output = output[:self.max_output_chars] + "\n... (truncated)"
                return {
                    "success": True,
                    "stdout": output,
                    "stderr": "",
                    "error": None,
                    "submitted": None,
                }

            # Handle error
            if "error" in msg:
                if msg.get("id") is not None and msg.get("id") != execute_request_id:
                    return {
                        "success": False,
                        "stdout": "",
                        "stderr": "",
                        "error": f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}",
                        "submitted": None,
                    }

                error = msg["error"]
                error_message = error.get("message", "Unknown error")
                error_data = error.get("data", {})
                error_type = error_data.get("type", "Error")

                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "error": f"{error_type}: {error_data.get('args') or error_message}",
                    "submitted": None,
                }

            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"Unexpected message from sandbox: {msg}",
                "submitted": None,
            }

    async def start(self) -> None:
        """Pre-warm the sandbox by starting the Deno process."""
        await self._ensure_process()

    async def stop(self) -> None:
        """Stop the Deno process and clean up resources."""
        if self._deno_process and self._deno_process.returncode is None:
            try:
                shutdown_msg = _jsonrpc_notification("shutdown")
                self._deno_process.stdin.write((shutdown_msg + "\n").encode())
                await self._deno_process.stdin.drain()
                self._deno_process.stdin.close()
                await self._deno_process.wait()
            except Exception:
                self._deno_process.kill()

        self._deno_process = None
        self._tools_registered = False
        self._mounted_files = False
        self._owner_thread = None

    def get_namespace(self) -> Dict[str, Any]:
        """Get current namespace (not available for Deno interpreter)."""
        return {}

    def clear_namespace(self) -> None:
        """Clear namespace by restarting the process."""
        if self._deno_process:
            asyncio.create_task(self.stop())
