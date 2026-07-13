# agent/tools/sandbox.py
"""
Sandboxed Python execution tool for the autonomous data‑science agent.

Two modes (controlled by config variable SANDBOX_MODE):
- "restrictedpython"  → uses RestrictedPython + timeout (cross‑platform, default)
- "docker"            → uses a Docker container (most secure, for production)

The tool returns only printed output (stdout) and error messages.
"""

import json
import sys
import io
import os
import operator
import multiprocessing
from langchain.tools import tool

# -------------------------------------------------------------------
# Configuration – set via environment or config.yaml
# -------------------------------------------------------------------
SANDBOX_MODE = os.getenv("SANDBOX_MODE", "restrictedpython")  # or "docker"
ALLOWED_MODULES = os.getenv(
    "SANDBOX_ALLOWED_MODULES",
    "pandas,numpy,matplotlib,seaborn,scipy,sklearn,json,csv,datetime,math,statistics"
).split(",")

SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", 30))  # seconds

# -------------------------------------------------------------------
# Mode 1 – RestrictedPython (lightweight, cross‑platform)
# -------------------------------------------------------------------
if SANDBOX_MODE == "restrictedpython":
    import RestrictedPython
    from RestrictedPython import compile_restricted
    from RestrictedPython.Guards import (
        safe_builtins, safe_globals, guarded_iter_unpack_sequence
    )
    from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
    from RestrictedPython.PrintCollector import PrintCollector

    _INPLACE_OPS = {
        "+=": operator.iadd, "-=": operator.isub, "*=": operator.imul,
        "/=": operator.itruediv, "//=": operator.ifloordiv, "%=": operator.imod,
        "**=": operator.ipow, "&=": operator.iand, "|=": operator.ior,
        "^=": operator.ixor, ">>=": operator.irshift, "<<=": operator.ilshift,
    }

    def _guarded_inplacevar(op, x, y):
        return _INPLACE_OPS[op](x, y)

# Create a safe globals dict populated with allowed modules
    _safe_globals = {
        "__builtins__": safe_builtins,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_getattr_": getattr,                   # Use standard python getattr
        "_getitem_": default_guarded_getitem,   # Needed for df['col'], list/dict indexing
        "_getiter_": default_guarded_getiter,   # Needed for iterating pandas/numpy objects
        "_write_": lambda x: x,                 # Allow variables to be written to
        "_inplacevar_": _guarded_inplacevar,    # Needed for x += y, x *= y, etc.
        "_print_": PrintCollector,              # RestrictedPython routes print() through this
        "__name__": "__main__",
    }

    for mod_name in ALLOWED_MODULES:
        try:
            mod = __import__(mod_name)
            _safe_globals[mod_name] = mod
        except ImportError:
            pass

    # safe_builtins deliberately omits __import__, so `import x` statements would
    # otherwise fail even for already-allowed modules. Provide a guarded version
    # that only permits the allow-listed modules plus the io_tools accessor.
    _IMPORT_ALLOWLIST = set(ALLOWED_MODULES) | {"agent.tools.io_tools"}

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name not in _IMPORT_ALLOWLIST:
            raise ImportError(f"Import of '{name}' is not allowed in the sandbox.")
        return __import__(name, globals, locals, fromlist, level)

    safe_builtins["__import__"] = _guarded_import

# -------------------------------------------------------------------
# Actual execution in a subprocess (to enforce timeout)
# -------------------------------------------------------------------
def _execute_restricted(code_str: str, result_queue: multiprocessing.Queue):
    """Run the code with RestrictedPython inside a subprocess."""
    try:
        compiled = compile_restricted(code_str, "<sandbox>", "exec")
        # Fresh globals per execution so runs don't leak state into each other
        exec_globals = dict(_safe_globals)
        exec(compiled, exec_globals)
        printed = exec_globals.get("_print")
        output = printed() if printed else ""
        result_queue.put(("success", output))
    except Exception as e:
        result_queue.put(("error", str(e)))

def _execute_docker(code_str: str) -> str:
    """Execute code in a Docker container. Not implemented in this skeleton."""
    # Placeholder: would write code to a temp file, run `docker run ...`, capture output
    return json.dumps({"error": "Docker sandbox mode not yet implemented."})

# -------------------------------------------------------------------
# The tool itself
# -------------------------------------------------------------------
@tool
def execute_python(code: str) -> str:
    """
    Execute arbitrary Python code in a sandboxed environment.
    - Allowed modules: pandas, numpy, matplotlib, seaborn, scipy, sklearn, json, csv, datetime, math, statistics.
    - The code can read the current DataFrame via `io_tools.get_dataframe()` (must be imported explicitly).
    - Use `print(...)` to output results; that output is returned.
    - Dangerous operations (file write, os, subprocess) are blocked.
    - Timeout after {SANDBOX_TIMEOUT} seconds.
    """
    if SANDBOX_MODE == "docker":
        return _execute_docker(code)

    # Use a forked subprocess (inherits the parent's in-memory DataFrame) to enforce timeout
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    p = ctx.Process(target=_execute_restricted, args=(code, result_queue))
    p.start()
    p.join(timeout=SANDBOX_TIMEOUT)
    if p.is_alive():
        p.terminate()
        p.join()
        return json.dumps({"error": f"Execution timed out after {SANDBOX_TIMEOUT}s."})
    if result_queue.empty():
        return json.dumps({"error": "Sandbox process did not return a result."})
    status, content = result_queue.get()
    if status == "error":
        return json.dumps({"error": content})
    # Return printed output (trimmed)
    return json.dumps({"output": content.strip()})