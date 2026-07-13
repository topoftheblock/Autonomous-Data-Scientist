from agent.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

planner_prompt = ChatPromptTemplate.from_template("""
You are a senior data scientist. You see ONLY a summary of a dataset (no raw rows).
Based on this summary, create a step-by-step plan to autonomously:
- profile the dataset (understand distributions, missing values)
- clean the data (handle missing, encode, scale, drop irrelevant columns)
- perform at least two statistical tests (e.g., t-test, chi-square, correlation)
- create at least two visualisations (histogram, boxplot, heatmap)
- write a final markdown report

**Keep the plan short (aim for 8-12 steps total).** The agent has a limited step
budget, so profile columns in batches (e.g. "Profile all numeric columns" or
"Profile all columns using profile_all_columns") rather than one step per column,
unless a column looks unusually important or problematic.

Output a **JSON list of strings**, each a concise instruction for a data‑science agent.
Example: ["Profile all columns","Check for missing values","Impute missing 'age' with median","Run t-test comparing 'income' by 'education'","Create histogram of 'income'","Create correlation heatmap","Generate final report"]

Dataset summary:
{data_summary}

Plan (JSON list only):
""")

def planner_node(state: AgentState) -> dict:
    """Generate a step-by-step plan from the data summary."""
    prompt_value = planner_prompt.invoke({"data_summary": state["data_summary"]})
    response = planner_llm.invoke(prompt_value)
    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        plan = json.loads(clean_content)
        if not isinstance(plan, list):
            raise ValueError("Plan is not a list")
    except Exception:
        # Fallback: hard‑coded basic plan
        plan = ["Profile all columns", "Clean data", "Run statistical tests", "Generate report"]
    return {"plan": plan, "current_step_index": 0}