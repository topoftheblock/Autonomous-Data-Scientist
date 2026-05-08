from agent.state import AgentState
from agent.tools.io_tools import load_csv

def ingestion_node(state: AgentState) -> AgentState:
    """Load the CSV and store a summary string."""
    result = load_csv.invoke({"file_path": state["file_path"]})
    state["data_summary"] = result
    state["step_results"] = []   # reset
    state["plan"] = []
    state["current_step_index"] = 0
    state["error"] = ""
    return state

