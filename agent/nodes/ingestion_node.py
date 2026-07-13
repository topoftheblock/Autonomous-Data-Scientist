from agent.state import AgentState
from agent.tools.io_tools import load_csv

def ingestion_node(state: AgentState) -> dict:
    """Load the CSV and store a summary string."""
    result = load_csv.invoke({"file_path": state["file_path"]})
    return {
        "data_summary": result,
        "plan": [],
        "current_step_index": 0,
        "error": "",
    }
