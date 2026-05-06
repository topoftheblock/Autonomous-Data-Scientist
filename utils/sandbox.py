# agent/utils/sandbox.py
"""
Core sandbox execution utility.
Used by the `execute_python` tool and possibly other nodes.
Supports two backends:
- 'restrictedpython' → RestrictedPython + subprocess (default, cross‑platform)
- 'docker'           → stub for real container isolation

Usage:
    from agent.utils.sandbox import execute_code_safely
    result = execute_code_safely(
        code="print(df.head())",
        timeout=30,
        allowed_modules=["pandas","numpy"],
    )
    # result is a dict: {"output": "...", "error": "...", "timed_out": False}
"""

import sys
import io
import multiprocessing
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Backend: RestrictedPython
# -------------------------------------------------------------------
SAFE_GLOBALS = None  # will be built once

def _build_safe_globals(allowed_modules: List[str]):
    """Construct a safe globals dictionary with whitelisted modules."""
    try:
        from RestrictedPython import compile_restricted
        from RestrictedPython.Guards import (
            safe_builtins,
            guarded_iter_unpack_sequence,
            guarded_getattr,
            guarded_iter,
            guarded_print,
        )
    except ImportError:
        raise ImportError(
            "RestrictedPython is required for safe execution. "
            "Install it with: pip install RestrictedPython"
        )

    g = {
        "__builtins__": safe_builtins,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_getattr_": guarded_getattr,
        "_getiter_": guarded_iter,
        "_print_": guarded_print,
        "_write_": lambda x: None,  # disable attribute assignment
        "__name__": "__main__",
    }
    for mod_name in allowed_modules:
        try:
            g[mod_name] = __import__(mod_name)
        except ImportError:
            logger.warning("Sandbox: module '%s' not available, skipping.", mod_name)
    return g


def _execute_restricted(code: str, allowed_modules: List[str], result_queue: multiprocessing.Queue):
    """Runs RestrictedPython code and puts (status, output) into the queue."""
    global SAFE_GLOBALS
    if SAFE_GLOBALS is None:
        SAFE_GLOBALS = _build_safe_globals(allowed_modules)

    try:
        from RestrictedPython import compile_restricted
        compiled = compile_restricted(code, "<sandbox>", "exec")
        out = io.StringIO()
        sys.stdout = out
        exec(compiled, SAFE_GLOBALS)
        sys.stdout = sys.__stdout__
        result_queue.put(("success", out.getvalue()))
    except Exception as e:
        sys.stdout = sys.__stdout__
        result_queue.put(("error", str(e)))


# -------------------------------------------------------------------
# Backend: Docker (stub)
# -------------------------------------------------------------------
def _execute_docker(code: str, timeout: int) -> Dict[str, Optional[str]]:
    """Placeholder for Docker isolation. Not yet implemented."""
    return {
        "output": None,
        "error": "Docker sandbox mode not yet implemented.",
        "timed_out": False,
    }


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def execute_code_safely(
    code: str,
    timeout: int = 30,
    allowed_modules: Optional[List[str]] = None,
    sandbox_mode: str = "restrictedpython",
) -> Dict[str, Optional[str]]:
    """
    Execute arbitrary Python code in a safe sandbox.

    Args:
        code: Python source code to run.
        timeout: Seconds to allow before killing the process.
        allowed_modules: List of module names to expose inside the sandbox.
        sandbox_mode: "restrictedpython" or "docker".

    Returns:
        dict with keys:
            output (str): stdout produced by the code (if any)
            error (str or None): error message (if any)
            timed_out (bool): whether execution was aborted due to timeout
    """
    if sandbox_mode == "docker":
        return _execute_docker(code, timeout)

    if allowed_modules is None:
        allowed_modules = ["pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn",
                           "json", "csv", "datetime", "math", "statistics"]

    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_execute_restricted,
        args=(code, allowed_modules, result_queue),
    )
    p.start()
    p.join(timeout=timeout)

    timed_out = False
    if p.is_alive():
        p.terminate()
        p.join()
        timed_out = True
        return {"output": None, "error": f"Execution timed out after {timeout}s.", "timed_out": True}

    if result_queue.empty():
        return {"output": None, "error": "Sandbox process returned no result.", "timed_out": False}

    status, content = result_queue.get()
    if status == "error":
        return {"output": None, "error": content, "timed_out": False}

    # Success
    return {"output": content.strip(), "error": None, "timed_out": False}