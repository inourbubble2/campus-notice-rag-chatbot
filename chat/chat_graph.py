from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig

from chat.schema import RAGState
from chat.nodes.guardrail import guardrail_node
from chat.nodes.validate import validate_node
from chat.nodes.agent import agent_node
from chat.nodes.retry_node import retry_node
from chat.tools import retreat_announcements

# Tools setup
tools = [retreat_announcements]
tool_node = ToolNode(tools)

graph = StateGraph(RAGState)

graph.add_node("guardrail", guardrail_node)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("validate", validate_node)
graph.add_node("retry", retry_node)

graph.add_edge(START, "guardrail")

def guardrail_router(state: RAGState):
    if state.guardrail and state.guardrail.policy == "BLOCK":
        return END
    return "agent"

graph.add_conditional_edges("guardrail", guardrail_router, ["agent", END])

graph.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END: "validate"} 
)

graph.add_edge("tools", "agent")

def validate_router(state: RAGState, config: RunnableConfig):
    max_retries = config.get("configurable", {}).get("max_retries", 2)
    
    if state.validation and state.validation.decision == "RETRY" and state.attempt < max_retries:
        return "retry"
    return END

graph.add_conditional_edges("validate", validate_router, ["retry", END])
graph.add_edge("retry", "agent")

app = graph.compile(checkpointer=MemorySaver())
