replan_prompt = ChatPromptTemplate.from_template("""
You previously made this plan:
{plan}

The execution failed at step "{failed_step}" with error:
{error}

Execution history so far:
{history}

Now create a **new, corrected plan** only for the remaining tasks (do NOT include already completed steps).
Output a JSON list of strings.
""")

def replan_node(state: AgentState) -> AgentState:
    """Adapt the remaining plan after a failure."""
    failed_idx = state["current_step_index"]
    failed_step = state["plan"][failed_idx] if failed_idx < len(state["plan"]) else "unknown"
    history = "\n".join(state["step_results"]) if state["step_results"] else "None"

    prompt_value = replan_prompt.invoke({
        "plan": json.dumps(state["plan"]),
        "failed_step": failed_step,
        "error": state["error"],
        "history": history
    })
    response = planner_llm.invoke(prompt_value)
    try:
        new_plan = json.loads(response.content)
        if not isinstance(new_plan, list):
            raise ValueError("Invalid plan format")
        # Replace the tail of the plan after the failed index
        state["plan"] = state["plan"][:failed_idx] + new_plan
        state["error"] = ""
    except Exception:
        # If even replan fails, abort by reducing plan
        state["plan"] = state["plan"][:failed_idx] + ["Abort and generate report with available results"]
    return state