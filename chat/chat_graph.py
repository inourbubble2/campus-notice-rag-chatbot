# chat_graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from chat.schema import RAGState
from chat.nodes.guardrail import guardrail_node
from chat.nodes.rewrite import rewrite_node
from chat.nodes.retrieve import retrieve_node
from chat.nodes.generate import generate_node
from chat.nodes.validate import validate_node

# =========================
# Graph Wiring (with loop)
# =========================
MAX_RETRY_ATTEMPTS = 3

graph = StateGraph(RAGState)

graph.add_node("guardrail", guardrail_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("validate", validate_node)

graph.add_edge(START, "guardrail")

# guardrail 결과에 따라 분기
def guardrail_router(state: RAGState):
    if (state.get("guardrail") or {}).get("policy") == "BLOCK":
        return END
    return "rewrite"

graph.add_conditional_edges("guardrail", guardrail_router, ["rewrite", END])
graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "validate")

# validate 결과에 따라 PASS면 종료, RETRY면 루프(재검색→재생성)
def validate_router(state: RAGState):
    if (state.get("validate") or {}).get("decision") == "RETRY" and state.get("attempt", 0) < MAX_RETRY_ATTEMPTS:
        return "retrieve"
    return END

graph.add_conditional_edges("validate", validate_router, ["retrieve", END])

app = graph.compile(checkpointer=MemorySaver())
