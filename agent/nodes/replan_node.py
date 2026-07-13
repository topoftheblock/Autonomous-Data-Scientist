import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from agent.state import AgentState

# We also need to import the LLM from your planner node so it can generate the new plan
from agent.nodes.planner_node import planner_llm

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

logger = logging.getLogger("datascience_agent")

def replan_node(state: AgentState) -> dict:
    """Adapt the remaining plan after a failure."""
    failed_idx = state["current_step_index"]
    failed_step = state["plan"][failed_idx] if failed_idx < len(state["plan"]) else "unknown"

    # Only take the last 5 steps to stay under token limits
    last_steps = state["step_results"][-5:]
    history = "\n".join(last_steps) if last_steps else "None"

    prompt_value = replan_prompt.invoke({
        "plan": json.dumps(state["plan"]),
        "failed_step": failed_step,
        "error": state["error"],
        "history": history
    })
    response = planner_llm.invoke(prompt_value)
    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        new_plan = json.loads(clean_content)
        if not isinstance(new_plan, list):
            raise ValueError("Invalid plan format")
        # Replace the tail of the plan after the failed index
        updated_plan = state["plan"][:failed_idx] + new_plan
    except Exception as e:
        logger.warning(f"Replan failed to parse LLM response ({e}); raw content: {response.content[:500]!r}")
        # If even replan fails, abort by reducing plan
        updated_plan = state["plan"][:failed_idx] + ["Abort and generate report with available results"]
    return {"plan": updated_plan, "error": ""}