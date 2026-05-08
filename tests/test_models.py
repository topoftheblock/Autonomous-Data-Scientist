# tests/test_nodes.py
"""
Pytest tests for the LangGraph nodes (planner, executor, replan, reporter, ingestion).
Run from the project root:
    pytest tests/test_nodes.py -v
"""

import json
import sys
import os
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.state import AgentState
from agent.nodes import ingestion_node, executor, planner_node, replan_node, reporter_node

# -------------------------------------------------------------------
# Helper: a minimal state fixture
# -------------------------------------------------------------------
@pytest.fixture
def base_state():
    return AgentState(
        file_path="dummy.csv",
        data_summary='{"shape": [100,5], "columns": ["age","income","gender"]}',
        data_parquet_path="/tmp/dummy.parquet",
        plan=[],
        current_step_index=0,
        step_results=[],
        error="",
        final_report="",
    )

# -------------------------------------------------------------------
# Ingestion node
# -------------------------------------------------------------------
@patch("agent.nodes.ingestion.load_csv")
def test_ingestion_node(mock_load, base_state):
    mock_load.invoke.return_value = '{"shape": [100,5]}'
    result = ingestion_node.ingestion_node(base_state)
    assert result["data_summary"] == '{"shape": [100,5]}'
    assert result["plan"] == []
    assert result["current_step_index"] == 0
    assert result["error"] == ""

# -------------------------------------------------------------------
# Planner node
# -------------------------------------------------------------------
@patch("agent.nodes.planner.planner_llm")
def test_planner_node_valid_plan(mock_llm, base_state):
    """Planner returns a valid JSON list -> plan is set."""
    mock_response = MagicMock()
    mock_response.content = json.dumps([
        "Profile column 'age'",
        "Profile column 'income'",
        "Generate report"
    ])
    mock_llm.invoke.return_value = mock_response

    result = planner_node.planner_node(base_state)
    assert len(result["plan"]) == 3
    assert result["current_step_index"] == 0
    assert result["plan"][0] == "Profile column 'age'"

@patch("agent.nodes.planner.planner_llm")
def test_planner_node_invalid_plan_fallback(mock_llm, base_state):
    """LLM returns garbage -> fallback plan is used."""
    mock_response = MagicMock()
    mock_response.content = "not a json list"
    mock_llm.invoke.return_value = mock_response

    result = planner_node.planner_node(base_state)
    # Should fall back to a hardcoded plan
    assert isinstance(result["plan"], list)
    assert len(result["plan"]) > 0
    assert result["error"] == ""

# -------------------------------------------------------------------
# Executor node
# -------------------------------------------------------------------
def test_executor_step_success(base_state, monkeypatch):
    """Simulate a successful step execution that returns DONE."""
    base_state["plan"] = ["Profile column 'age'"]
    base_state["current_step_index"] = 0

    # Mock the agent executor
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {"output": "DONE\nProfiled age: mean 35, missing 0."}
    monkeypatch.setattr(executor, "AgentExecutor", lambda *args, **kwargs: mock_executor)

    result = executor.execute_step(base_state)
    assert result["error"] == ""
    assert len(result["step_results"]) == 1
    assert "Step 1" in result["step_results"][0]
    assert base_state["current_step_index"] == 1

def test_executor_step_failure(base_state, monkeypatch):
    """Simulate a step that raises an exception -> error set."""
    base_state["plan"] = ["Bad step"]
    base_state["current_step_index"] = 0

    mock_executor = MagicMock()
    mock_executor.invoke.side_effect = Exception("Tool not found")
    monkeypatch.setattr(executor, "AgentExecutor", lambda *args, **kwargs: mock_executor)

    result = executor.execute_step(base_state)
    assert "Tool not found" in result["error"]
    assert base_state["current_step_index"] == 0  # not incremented

def test_executor_skips_if_error_already_set(base_state):
    """If error is already present, executor does nothing."""
    base_state["error"] = "previous error"
    base_state["plan"] = ["step"]
    base_state["current_step_index"] = 0
    result = executor.execute_step(base_state)
    assert result["error"] == "previous error"
    assert result["step_results"] == []

# -------------------------------------------------------------------
# Replan node
# -------------------------------------------------------------------
@patch("agent.nodes.replan.planner_llm")
def test_replan_node_replaces_failed_step(mock_llm, base_state):
    """Replan generates a new plan for remaining steps."""
    base_state["plan"] = ["Profile age", "Profile income", "Failing step", "Generate report"]
    base_state["current_step_index"] = 2  # failed at index 2
    base_state["error"] = "Column not found"
    base_state["step_results"] = ["Step 1: Done", "Step 2: Done"]

    mock_response = MagicMock()
    mock_response.content = json.dumps(["Alternative step", "Generate report"])
    mock_llm.invoke.return_value = mock_response

    result = replan_node.replan_node(base_state)
    assert result["error"] == ""
    assert result["plan"] == ["Profile age", "Profile income", "Alternative step", "Generate report"]
    assert base_state["plan"] == ["Profile age", "Profile income", "Alternative step", "Generate report"]

@patch("agent.nodes.replan.planner_llm")
def test_replan_node_fallback_when_llm_fails(mock_llm, base_state):
    """If replan LLM fails, a fallback abort step is added."""
    base_state["plan"] = ["Step1", "Step2"]
    base_state["current_step_index"] = 1
    base_state["error"] = "some error"
    mock_response = MagicMock()
    mock_response.content = "garbage"
    mock_llm.invoke.return_value = mock_response

    result = replan_node.replan_node(base_state)
    assert "Abort" in result["plan"][-1]

# -------------------------------------------------------------------
# Report node
# -------------------------------------------------------------------
def test_report_node(base_state):
    base_state["step_results"] = ["Step 1: Profiled age", "Step 2: Cleaned data"]
    result = reporter_node.report_node(base_state)
    assert "# Autonomous Data Science Report" in result["final_report"]
    assert "Step 1" in result["final_report"]
    assert os.path.exists("report.md")  # file written

# -------------------------------------------------------------------
# Conditional edge logic (after_execution)
# -------------------------------------------------------------------
def test_after_execution_routes_correctly():
    from agent.graph import after_execution

    # All steps done -> report
    state = AgentState(
        file_path="", data_summary="", plan=["a","b"], current_step_index=2,
        step_results=[], error="", final_report=""
    )
    assert after_execution(state) == "report"

    # More steps -> executor
    state["current_step_index"] = 1
    assert after_execution(state) == "executor"

    # Error -> replan
    state["error"] = "failed"
    assert after_execution(state) == "replan"