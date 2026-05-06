from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    file_path: str
    data_summary: str
    plan: List[str]                     # high-level steps
    current_step_index: int
    step_results: Annotated[List[str], operator.add]  # accumulate results
    error: str                          # non-empty if a step failed
    final_report: str