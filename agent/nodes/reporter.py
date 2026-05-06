def report_node(state: AgentState) -> AgentState:
    """Compile all step outputs into a final report."""
    history = "\n\n".join(
        f"### Step {i+1}\n{res}" for i, res in enumerate(state["step_results"])
    )
    report = f"# Autonomous Data Science Report\n\n## Execution Log\n\n{history}"
    state["final_report"] = report
    # Optionally, write to file:
    with open("report.md", "w") as f:
        f.write(report)
    return state