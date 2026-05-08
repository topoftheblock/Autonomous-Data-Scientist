from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.ingestion_node import ingestion_node
from agent.nodes.planner_node import planner_node
from agent.nodes.execution_step import execute_step
from agent.nodes.replan_node import replan_node
from agent.nodes.report_node import reporter_node

graph = StateGraph(AgentState)
graph.add_node("ingest", ingestion_node)
# Add nodes
graph.add_node("ingest", ingestion_node)
graph.add_node("planner", planner_node)
graph.add_node("executor", execute_step)
graph.add_node("replan", replan_node)
graph.add_node("report", report_node)

# Edges
graph.set_entry_point("ingest")
graph.add_edge("ingest", "planner")
graph.add_edge("planner", "executor")

# After execution, decide where to go next
def after_execution(state: AgentState) -> str:
    if state["error"]:
        return "replan"
    elif state["current_step_index"] < len(state["plan"]):
        return "executor"   # loop to do next step
    else:
        return "report"

graph.add_conditional_edges("executor", after_execution, {
    "executor": "executor",
    "replan": "replan",
    "report": "report",
})

# After replan, go back to executor
graph.add_edge("replan", "executor")

# Report is the end
graph.add_edge("report", END)

# Compile
app = graph.compile()