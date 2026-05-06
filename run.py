initial_state = AgentState(
    file_path="./data/sales_data.csv",
    data_summary="",
    plan=[],
    current_step_index=0,
    step_results=[],
    error="",
    final_report="",
)

final_state = app.invoke(initial_state)
print("Final Report:\n", final_state["final_report"])