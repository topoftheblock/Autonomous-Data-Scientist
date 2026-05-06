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

    # Create a safe globals dict populated with allowed modules
    _safe_globals = {
        "__builtins__": safe_builtins,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_getattr_": RestrictedPython.Guards.guarded_getattr,
        "_write_": lambda x: None,               # disable attribute writing
        "_getiter_": RestrictedPython.Guards.guarded_iter,
        "_print_": RestrictedPython.Guards.guarded_print,
        "__name__": "__main__",
    }
    for mod_name in ALLOWED_MODULES:
        try:
            mod = __import__(mod_name)
            _safe_globals[mod_name] = mod
        except ImportError:
            pass

# -------------------------------------------------------------------
# Actual execution in a subprocess (to enforce timeout)
# -------------------------------------------------------------------
def _execute_restricted(code_str: str, result_queue: multiprocessing.Queue):
    """Run the code with RestrictedPython inside a subprocess."""
    try:
        compiled = compile_restricted(code_str, "<sandbox>", "exec")
        # Redirect stdout
        out = io.StringIO()
        sys.stdout = out
        exec(compiled, _safe_globals)
        sys.stdout = sys.__stdout__
        result_queue.put(("success", out.getvalue()))
    except Exception as e:
        sys.stdout = sys.__stdout__
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
    - The code can read the current DataFrame using `io_tools._DF` (must be imported explicitly).
    - Use `print(...)` to output results; that output is returned.
    - Dangerous operations (file write, os, subprocess) are blocked.
    - Timeout after {SANDBOX_TIMEOUT} seconds.
    """
    if SANDBOX_MODE == "docker":
        return _execute_docker(code)

    # Use a subprocess to enforce timeout
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_execute_restricted, args=(code, result_queue))
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