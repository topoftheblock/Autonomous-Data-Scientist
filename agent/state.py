# agent/state.py
from typing import TypedDict, List, Annotated, Optional
import operator

class AgentState(TypedDict):
    """State shared across all nodes in the LangGraph pipeline."""
    
    # Input
    file_path: str                     # Path to the CSV/Excel file to analyse
    
    # After ingestion
    data_summary: str                  # Structured summary (JSON string) from load_and_inspect
    data_parquet_path: Optional[str]   # Path to cleaned data in Parquet format (for passing between nodes)
    
    # Planning
    plan: List[str]                    # Ordered list of step descriptions (e.g., "Profile column 'age'")
    current_step_index: int            # Index of the currently executing (or next) plan step
    
    # Execution history
    # Annotated[List, operator.add] means that whenever a node returns a dict with
    # "step_results": ["..."] , the string gets appended to the existing list.
    step_results: Annotated[List[str], operator.add]
    
    # Error handling
    error: str                         # Non‑empty if the last step failed; triggers replanning
    
    # Final output
    final_report: str                  # Markdown report content
    statistical_tests: Annotated[List[dict], operator.add]   # append‑only list
    plot_paths: Annotated[List[str], operator.add]
    correlation_matrix: Optional[dict]
    cleaning_summary: Optional[dict]