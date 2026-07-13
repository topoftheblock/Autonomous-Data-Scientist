# tests/test_graph.py
"""
Integration test for the full LangGraph pipeline.

Runs the graph end‑to‑end with a tiny CSV and mocked LLM responses.
Verifies:
- The plan is generated and executed
- Step results are accumulated
- No errors occur
- Final report is produced

Run from project root:
    pytest tests/test_graph.py -v
"""

import json
import sys
import os
import tempfile
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.state import AgentState
from agent.graph import app   # the compiled StateGraph

# ---------------------------------------------------------------
# Helper: create a tiny CSV for ingestion
# ---------------------------------------------------------------
@pytest.fixture
def sample_csv():
    df = pd.DataFrame({
        "age": [25, 30, 22],
        "income": [50000, 60000, 75000],
        "city": ["NY", "LA", "NY"],
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f, index=False)
    yield f.name
    os.unlink(f.name)   # cleanup

# ---------------------------------------------------------------
# Test the whole graph with mocked LLMs
# ---------------------------------------------------------------
@patch("agent.nodes.execution_step.AgentExecutor")
@patch("agent.nodes.planner_node.planner_llm")
def test_full_pipeline(mock_planner_llm, mock_agent_executor_cls, sample_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # ---- Mock planner LLM -------------------------------------------------
    mock_response = MagicMock()
    mock_response.content = json.dumps([
        "Profile column 'age'",
        "Profile column 'income'",
        "Profile column 'city'",
        "Generate report"
    ])
    mock_planner_llm.invoke.return_value = mock_response

    # ---- Mock executor AgentExecutor (one call per plan step) ------------
    # The executor node creates a new AgentExecutor instance each time.
    # We'll return a mock that simulates a successful "DONE" output.
    step_outputs = [
        {"output": "DONE\nProfiled age: no missing, mean=25.7"},
        {"output": "DONE\nProfiled income: no missing, mean=61666.0"},
        {"output": "DONE\nProfiled city: categorical, 3 unique"},
        {"output": "DONE\nGenerated final report"},
    ]

    def create_mock_executor(*args, **kwargs):
        mock_exec = MagicMock()
        mock_exec.invoke.return_value = step_outputs.pop(0) if step_outputs else {"output": "DONE"}
        return mock_exec

    mock_agent_executor_cls.side_effect = create_mock_executor

    # ---- Run the graph ---------------------------------------------------
    initial_state = AgentState(
        file_path=sample_csv,
        data_summary="",
        data_parquet_path=None,
        plan=[],
        current_step_index=0,
        step_results=[],
        error="",
        final_report="",
    )

    # Run with recursion limit high enough for all steps
    final_state = app.invoke(initial_state, {"recursion_limit": 20})

    # ---- Assertions -----------------------------------------------------
    # No errors
    assert final_state["error"] == ""

    # Plan was set
    assert len(final_state["plan"]) == 4

    # All 4 plan steps executed, plus one entry logged by the reporter itself
    assert len(final_state["step_results"]) == 5

    # Each step result contains the summary
    assert "Profiled age" in final_state["step_results"][0]
    assert "Profiled income" in final_state["step_results"][1]
    assert "Profiled city" in final_state["step_results"][2]
    assert "Generated final report" in final_state["step_results"][3]

    # Final report exists and contains step results
    assert "# Data Science Report" in final_state["final_report"]
    assert "Step 1" in final_state["final_report"]
    assert "Profiled age" in final_state["final_report"]

    # Data summary was populated by ingestion
    assert "shape" in final_state["data_summary"]

    # Clean up any files created by the reporter
    if os.path.exists("report.md"):
        os.unlink("report.md")


# ---------------------------------------------------------------
# Test error recovery (replan) path
# ---------------------------------------------------------------
@patch("agent.nodes.execution_step.AgentExecutor")
@patch("agent.nodes.replan_node.planner_llm")
@patch("agent.nodes.planner_node.planner_llm")
def test_error_replan_flow(mock_init_planner, mock_replan_llm, mock_executor_cls, sample_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Initial planner returns a plan that will fail on step 2
    mock_response = MagicMock()
    mock_response.content = json.dumps([
        "Profile column 'age'",
        "Run failing step",
        "Generate report"
    ])
    mock_init_planner.invoke.return_value = mock_response

    # Executor mocks: 1st call succeeds, 2nd call fails, subsequent calls (post-replan) succeed
    call_count = {"n": 0}

    class FakeExecutor:
        def invoke(self, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise Exception("Tool broken")
            return {"output": f"DONE\nStep {call_count['n']} done."}

    mock_executor_cls.side_effect = lambda *a, **kw: FakeExecutor()

    # Replan LLM will be called to fix the plan
    mock_replan_response = MagicMock()
    mock_replan_response.content = json.dumps(["Alternative clean step", "Generate report"])
    mock_replan_llm.invoke.return_value = mock_replan_response

    # Run graph
    initial_state = AgentState(
        file_path=sample_csv,
        data_summary="",
        data_parquet_path=None,
        plan=[],
        current_step_index=0,
        step_results=[],
        error="",
        final_report="",
    )
    final_state = app.invoke(initial_state, {"recursion_limit": 20})

    # After failure + replan, the pipeline should recover and finish
    assert final_state["error"] == ""
    # Plan should have been updated to replace failing step
    assert "Alternative clean step" in final_state["plan"]
    # At least 1 step succeeded, then replanned steps executed
    assert len(final_state["step_results"]) >= 2
    # Final report present
    assert "# Data Science Report" in final_state["final_report"]

    # Cleanup
    if os.path.exists("report.md"):
        os.unlink("report.md")