// Synalinks REPL Runner - Deno/Pyodide Sandbox
// Secure Python execution via WebAssembly

// Deno 2.x requires npm packages to be configured in deno.json
// with nodeModulesDir: "auto"
import pyodideModule from "pyodide/pyodide.js";

// Async line reader for stdin - handles partial line buffering correctly
async function* readLines(reader) {
  const decoder = new TextDecoder();
  let partialLine = '';

  for await (const chunk of reader) {
    const text = decoder.decode(chunk);
    const lines = (partialLine + text).split('\n');

    // Keep the last element as partial (may be empty or incomplete)
    partialLine = lines.pop();

    // Yield all complete lines (skip empty lines)
    for (const line of lines) {
      if (line.length > 0) {
        yield line;
      }
    }
  }

  // Yield any remaining partial line at EOF
  if (partialLine.length > 0) {
    yield partialLine;
  }
}

// =============================================================================
// Python Code Templates
// =============================================================================

const PYTHON_SETUP_CODE = `
import sys, io, json
old_stdout, old_stderr = sys.stdout, sys.stderr
buf_stdout, buf_stderr = io.StringIO(), io.StringIO()
sys.stdout, sys.stderr = buf_stdout, buf_stderr

def last_exception_args():
    return json.dumps(sys.last_exc.args) if sys.last_exc else None

class FinalOutput(BaseException):
    # Control-flow exception to signal completion (like StopIteration)
    pass

# Default SUBMIT for single-output signatures
if 'SUBMIT' not in dir():
    def SUBMIT(**kwargs):
        raise FinalOutput(kwargs)
`;

// Convert JavaScript value to Python literal syntax
const toPythonLiteral = (value) => {
  if (value === null) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  return JSON.stringify(value);
};

// Generate a tool wrapper function with typed signature
const makeToolWrapper = (toolName, parameters = []) => {
  const sigParts = parameters.map(p => {
    let part = p.name;
    if (p.type) part += `: ${p.type}`;
    if (p.default !== undefined) part += ` = ${toPythonLiteral(p.default)}`;
    return part;
  });
  const signature = sigParts.join(', ');
  const argNames = parameters.map(p => p.name);

  if (parameters.length === 0) {
    return `
import json
from pyodide.ffi import run_sync, JsProxy
def ${toolName}(*args, **kwargs):
    result = run_sync(_js_tool_call("${toolName}", json.dumps({"args": args, "kwargs": kwargs})))
    return result.to_py() if isinstance(result, JsProxy) else result
`;
  }

  return `
import json
from pyodide.ffi import run_sync, JsProxy
def ${toolName}(${signature}):
    _args = [${argNames.join(', ')}]
    result = run_sync(_js_tool_call("${toolName}", json.dumps({"args": _args, "kwargs": {}})))
    return result.to_py() if isinstance(result, JsProxy) else result
`;
};

// Generate SUBMIT function with output field signature
const makeSubmitWrapper = (outputs) => {
  if (!outputs || outputs.length === 0) {
    return `
def SUBMIT(**kwargs):
    raise FinalOutput(kwargs)
`;
  }

  const sigParts = outputs.map(o => {
    let part = o.name;
    if (o.type) part += `: ${o.type}`;
    return part;
  });
  const dictParts = outputs.map(o => `"${o.name}": ${o.name}`);

  return `
def SUBMIT(${sigParts.join(', ')}):
    raise FinalOutput({${dictParts.join(', ')}})
`;
};

// =============================================================================
// JSON-RPC 2.0 Helpers
// =============================================================================

const JSONRPC_PROTOCOL_ERRORS = {
  ParseError: -32700,
  InvalidRequest: -32600,
  MethodNotFound: -32601,
};

const JSONRPC_APP_ERRORS = {
  SyntaxError: -32000,
  NameError: -32001,
  TypeError: -32002,
  ValueError: -32003,
  AttributeError: -32004,
  IndexError: -32005,
  KeyError: -32006,
  RuntimeError: -32007,
  CodeInterpreterError: -32008,
  Unknown: -32099,
};

const jsonrpcRequest = (method, params, id) =>
  JSON.stringify({ jsonrpc: "2.0", method, params, id });

const jsonrpcNotification = (method, params = null) => {
  const msg = { jsonrpc: "2.0", method };
  if (params) msg.params = params;
  return JSON.stringify(msg);
};

const jsonrpcResult = (result, id) =>
  JSON.stringify({ jsonrpc: "2.0", result, id });

const jsonrpcError = (code, message, id, data = null) => {
  const err = { code, message };
  if (data) err.data = data;
  return JSON.stringify({ jsonrpc: "2.0", error: err, id });
};

// Global handler for uncaught promise rejections
globalThis.addEventListener("unhandledrejection", (event) => {
  event.preventDefault();
  console.log(jsonrpcError(
    JSONRPC_APP_ERRORS.RuntimeError,
    `Unhandled async error: ${event.reason?.message || event.reason}`,
    null
  ));
});

// =============================================================================
// Main
// =============================================================================

const pyodide = await pyodideModule.loadPyodide();

// Tool call support
const stdinReader = readLines(Deno.stdin.readable);
let requestIdCounter = 0;

// Bridge function for Python to call host-side tools
async function toolCallBridge(name, argsJson) {
  const requestId = `tc_${Date.now()}_${++requestIdCounter}`;

  try {
    const parsedArgs = JSON.parse(argsJson);

    console.log(jsonrpcRequest("tool_call", {
      name: name,
      args: parsedArgs.args || [],
      kwargs: parsedArgs.kwargs || {}
    }, requestId));

    const { value: responseLine, done } = await stdinReader.next();
    if (done) {
      throw new Error("stdin closed while waiting for tool response");
    }

    const response = JSON.parse(responseLine);

    if (response.id !== requestId) {
      throw new Error(`Unexpected response: expected id ${requestId}, got ${response.id}`);
    }

    if (response.error) {
      throw new Error(response.error.message || "Tool call failed");
    }

    const result = response.result;
    if (result.type === "json") {
      return JSON.parse(result.value);
    }
    return result.value;
  } catch (error) {
    throw new Error(`Tool bridge error for '${name}': ${error.message}`);
  }
}

pyodide.globals.set("_js_tool_call", toolCallBridge);

// Load environment variables if provided
try {
  const env_vars = (Deno.args[0] ?? "").split(",").filter(Boolean);
  for (const key of env_vars) {
    const val = Deno.env.get(key);
    if (val !== undefined) {
      pyodide.runPython(`
import os
os.environ[${JSON.stringify(key)}] = ${JSON.stringify(val)}
      `);
    }
  }
} catch (e) {
  console.error("Error setting environment variables:", e);
}

// Main message loop
while (true) {
  const { value: line, done } = await stdinReader.next();
  if (done) break;

  let input;
  try {
    input = JSON.parse(line);
  } catch (error) {
    console.log(jsonrpcError(JSONRPC_PROTOCOL_ERRORS.ParseError, "Invalid JSON: " + error.message, null));
    continue;
  }

  if (typeof input !== 'object' || input === null || input.jsonrpc !== "2.0") {
    console.log(jsonrpcError(JSONRPC_PROTOCOL_ERRORS.InvalidRequest, "Invalid JSON-RPC 2.0 message", null));
    continue;
  }

  const method = input.method;
  const params = input.params || {};
  const requestId = input.id;

  // Handle notifications
  if (method === "sync_file") {
    try {
      const virtualPath = params.virtual_path;
      const hostPath = params.host_path || virtualPath;
      await Deno.writeFile(hostPath, pyodide.FS.readFile(virtualPath));
    } catch (e) { /* ignore */ }
    continue;
  }

  if (method === "shutdown") break;

  // Handle requests
  if (method === "mount_file") {
    const hostPath = params.host_path;
    const virtualPath = params.virtual_path || hostPath;
    try {
      const contents = await Deno.readFile(hostPath);
      const dirs = virtualPath.split('/').slice(1, -1);
      let cur = '';
      for (const d of dirs) {
        cur += '/' + d;
        try {
          pyodide.FS.mkdir(cur);
        } catch (e) {
          if (!e.message?.includes('File exists')) throw e;
        }
      }
      pyodide.FS.writeFile(virtualPath, contents);
      console.log(jsonrpcResult({ mounted: virtualPath }, requestId));
    } catch (e) {
      console.log(jsonrpcError(JSONRPC_APP_ERRORS.RuntimeError, `Mount failed: ${e.message}`, requestId));
    }
    continue;
  }

  if (method === "register") {
    const toolNames = [];

    if (params.tools) {
      for (const tool of params.tools) {
        if (typeof tool === 'string') {
          pyodide.runPython(makeToolWrapper(tool, []));
          toolNames.push(tool);
        } else {
          pyodide.runPython(makeToolWrapper(tool.name, tool.parameters || []));
          toolNames.push(tool.name);
        }
      }
    }

    if (params.outputs) {
      pyodide.runPython(makeSubmitWrapper(params.outputs));
    }

    console.log(jsonrpcResult({
      tools: toolNames,
      outputs: params.outputs ? params.outputs.map(o => o.name) : []
    }, requestId));
    continue;
  }

  if (method === "execute") {
    const code = params.code || "";
    let setupCompleted = false;

    try {
      await pyodide.loadPackagesFromImports(code);
      pyodide.runPython(PYTHON_SETUP_CODE);
      setupCompleted = true;

      const result = await pyodide.runPythonAsync(code);
      const capturedStdout = pyodide.runPython("buf_stdout.getvalue()");

      let output = (result === null || result === undefined)
        ? capturedStdout
        : (result.toJs?.() ?? result);
      console.log(jsonrpcResult({ output }, requestId));
    } catch (error) {
      const errorType = error.type || "Error";
      const errorMessage = (error.message || "").trim();

      // FinalOutput is success, not error
      if (errorType === "FinalOutput") {
        const last_exception_args = pyodide.globals.get("last_exception_args");
        const errorArgs = JSON.parse(last_exception_args()) || [];
        const answer = errorArgs[0] || null;
        console.log(jsonrpcResult({ final: answer }, requestId));
        continue;
      }

      let errorArgs = [];
      if (errorType !== "SyntaxError") {
        const last_exception_args = pyodide.globals.get("last_exception_args");
        errorArgs = JSON.parse(last_exception_args()) || [];
      }

      const errorCode = JSONRPC_APP_ERRORS[errorType] || JSONRPC_APP_ERRORS.Unknown;
      console.log(jsonrpcError(errorCode, errorMessage, requestId, { type: errorType, args: errorArgs }));
    } finally {
      if (setupCompleted) {
        try {
          pyodide.runPython("sys.stdout, sys.stderr = old_stdout, old_stderr");
        } catch (e) { /* ignore */ }
      }
    }
    continue;
  }

  console.log(jsonrpcError(JSONRPC_PROTOCOL_ERRORS.MethodNotFound, `Method not found: ${method}`, requestId));
}
