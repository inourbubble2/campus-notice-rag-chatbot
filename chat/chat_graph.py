from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from chat.schema import RAGState
from chat.nodes.guardrail import guardrail_node
from chat.nodes.rewrite import rewrite_node
from chat.nodes.retrieve import retrieve_node
from chat.nodes.generate import generate_node
from chat.nodes.validate import validate_node
from chat.nodes.refine_query import refine_query_node

graph = StateGraph(RAGState)

graph.add_node("guardrail", guardrail_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("validate", validate_node)
graph.add_node("refine_query", refine_query_node)

graph.add_edge(START, "guardrail")

def guardrail_router(state: RAGState):
    if state.guardrail and state.guardrail.policy == "BLOCK":
        return END
    return "rewrite"

graph.add_conditional_edges("guardrail", guardrail_router, ["rewrite", END])
graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "validate")

def validate_router(state: RAGState, config: RunnableConfig):
    max_retries = config.get("configurable", {}).get("max_retries", 3)
    
    if state.validation and state.validation.decision == "RETRY" and state.attempt < max_retries:
        return "refine_query"
    return END

graph.add_conditional_edges("validate", validate_router, ["refine_query", END])
graph.add_edge("refine_query", "retrieve")

app = graph.compile(checkpointer=MemorySaver())
