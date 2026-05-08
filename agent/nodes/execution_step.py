from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

executor_llm = ChatOpenAI(model="gpt-4o", temperature=0)

executor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an autonomous data‑science agent.
Your current task is: {step}
You have access to tools for data loading, profiling, cleaning, statistical tests, and plotting.
**Important:** After you complete the task, output exactly:
DONE
Followed by a brief summary of what you did."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Execute the task: {step}"),
])

def execute_step(state: AgentState) -> AgentState:
    """Execute the current plan step using an agent with tools."""
    if state.get("error"):
        # If we came here after an error, skip to replan
        return state

    plan = state["plan"]
    idx = state["current_step_index"]
    if idx >= len(plan):
        return state

    step = plan[idx]
    # Build agent for this step
    agent = create_openai_tools_agent(executor_llm, tools, executor_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True,
        max_iterations=10, handle_parsing_errors=True,
        return_intermediate_steps=False
    )

    # Chat history can contain previous step summaries
    chat_history = []
    for prev_result in state["step_results"]:
        chat_history.append(("ai", prev_result))

    try:
        result = agent_executor.invoke({
            "step": step,
            "chat_history": chat_history
        })
        output = result["output"]
        # Extract the part after "DONE" as the summary
        if "DONE" in output:
            summary = output.split("DONE", 1)[1].strip()
        else:
            summary = output
        state["step_results"].append(f"Step {idx+1} ({step}): {summary}")
        state["current_step_index"] += 1
        state["error"] = ""
    except Exception as e:
        state["error"] = f"Step '{step}' failed with: {str(e)}"
    return state