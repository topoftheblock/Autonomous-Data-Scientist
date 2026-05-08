from agent.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)

planner_prompt = ChatPromptTemplate.from_template("""
You are a senior data scientist. You see ONLY a summary of a dataset (no raw rows).
Based on this summary, create a detailed step-by-step plan to autonomously:
- profile every column (understand distributions, missing values)
- clean the data (handle missing, encode, scale, drop irrelevant columns)
- perform at least two statistical tests (e.g., t-test, chi-square, correlation)
- create at least two visualisations (histogram, boxplot, heatmap)
- write a final markdown report

Output a **JSON list of strings**, each a concise instruction for a data‑science agent.
Example: ["Profile column 'age'","Profile column 'income'","Impute missing 'age' with median","...","Run t-test comparing 'income' by 'education'","Create histogram of 'income'","Generate final report"]

Dataset summary:
{data_summary}

Plan (JSON list only):
""")

def planner_node(state: AgentState) -> AgentState:
    """Generate a step-by-step plan from the data summary."""
    prompt_value = planner_prompt.invoke({"data_summary": state["data_summary"]})
    response = planner_llm.invoke(prompt_value)
    try:
        plan = json.loads(response.content)
        if not isinstance(plan, list):
            raise ValueError("Plan is not a list")
        state["plan"] = plan
        state["current_step_index"] = 0
    except Exception as e:
        # Fallback: hard‑coded basic plan
        state["plan"] = ["Profile all columns", "Clean data", "Run statistical tests", "Generate report"]
    return state