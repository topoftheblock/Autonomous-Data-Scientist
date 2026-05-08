# agent/nodes/reporter.py
"""
Reporter node – compiles the final Markdown report.
"""

import json
import os
from agent.state import AgentState
from agent.tools.reporting import generate_report

def reporter_node(state: AgentState) -> dict:
    # Prepare data for the report tool
    report_data = {
        "file_path": state["file_path"],
        "data_summary": state.get("data_summary", ""),
        "step_results": state.get("step_results", []),
        # The following might have been stored earlier by the executor
        "statistical_tests": state.get("statistical_tests", []),
        "plot_paths": state.get("plot_paths", []),
        "correlation_matrix": state.get("correlation_matrix", {}),
        "cleaning_summary": state.get("cleaning_summary", {}),
    }

    # Call the report generation tool
    result_str = generate_report.invoke({"report_data_json": json.dumps(report_data)})
    result = json.loads(result_str)

    report_path = result.get("report_path", "")

    # Read the report content (Markdown) so we can store it in the state
    report_content = ""
    if report_path and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()

    return {
        "final_report": report_content,
        "step_results": [f"Formal report generated at {report_path}"],
    }