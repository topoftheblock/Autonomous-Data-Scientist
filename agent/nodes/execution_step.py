import logging
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 1. Add the missing LLM and State imports ---
from langchain_openai import ChatOpenAI
from agent.state import AgentState

# --- 2. Import all the tools the agent needs ---
from agent.tools.io_tools import load_csv, get_data_summary, save_dataframe, reload_dataframe
from agent.tools.profiling import profile_column, profile_all_columns, missing_summary
from agent.tools.cleaning import clean_data, get_cleaning_recommendations
from agent.tools.statistics import run_statistical_test, correlation_matrix, linear_regression
from agent.tools.visualization import plot_histogram, plot_boxplot, plot_scatter, plot_correlation_heatmap, plot_barplot, create_analysis_plots
from agent.tools.sandbox import execute_python

# --- 3. Combine them into the 'tools' list ---
tools = [
    load_csv, get_data_summary, save_dataframe, reload_dataframe,
    profile_column, profile_all_columns, missing_summary,
    clean_data, get_cleaning_recommendations,
    run_statistical_test, correlation_matrix, linear_regression,
    plot_histogram, plot_boxplot, plot_scatter, plot_correlation_heatmap, plot_barplot, create_analysis_plots,
    execute_python
]
executor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

executor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an autonomous data-science agent.
Your current task is: {step}
The dataset is located at: {file_path}
You currently have the file path. Do NOT ask for it. Use the load_csv tool immediately.
**CRITICAL RULE:** NEVER ask the user for input, clarifications, or file paths. Use your tools (like load_csv) to access the data automatically.
**Important:** After you complete the task, output exactly:
DONE
Followed by a brief summary of what you did."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Execute the task: {step}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

logger = logging.getLogger("datascience_agent")

def execute_step(state: AgentState) -> dict:
    """Execute the current plan step using an agent with tools."""
    if state.get("error"):
        # If we came here after an error, skip to replan
        return {}

    plan = state["plan"]
    idx = state["current_step_index"]
    if idx >= len(plan):
        return {}

    step = plan[idx]
    # Build agent for this step
    agent = create_openai_tools_agent(executor_llm, tools, executor_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True,
        max_iterations=10, handle_parsing_errors=True,
        return_intermediate_steps=False
    )

    # Chat history can contain previous step summaries
    chat_history = [("ai", prev_result) for prev_result in state["step_results"]]

    try:
        result = agent_executor.invoke({
            "step": step,
            "file_path": state["file_path"],
            "chat_history": chat_history
        })

        output = result["output"]
        # Extract the part after "DONE" as the summary
        if "DONE" in output:
            summary = output.split("DONE", 1)[1].strip()
            return {
                "step_results": [f"Step {idx+1} ({step}): {summary}"],
                "current_step_index": idx + 1,
                "error": "",
            }
        else:
            logger.warning(f"Step '{step}' did not signal DONE. Raw output: {output[:500]!r}")
            return {"error": "Agent did not signal completion with DONE."}
    except Exception as e:
        logger.warning(f"Step '{step}' failed with: {e}")
        return {"error": f"Step '{step}' failed with: {str(e)}"}