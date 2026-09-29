from langgraph.graph import StateGraph, START, END
from graph.nodes.retrieve import retrieve_relevant_docs_node
from graph.nodes.grade_documents import grade_retrieved_documents
from graph.nodes.web_search import web_search
from graph.nodes.generate import generate_node
from graph.state import GraphState


def decide_next_step(state: GraphState):
    """If the grader rejected any document, go search the web; otherwise generate the answer."""
    if state["web_search_flag"]:
        return "web_search"
    return "generate_node"


builder = StateGraph(GraphState)

builder.add_node("retrieve_relevant_docs_node",retrieve_relevant_docs_node)
builder.add_node("grade_retrieved_documents",grade_retrieved_documents)
builder.add_node("web_search",web_search)
builder.add_node("generate_node",generate_node)

builder.add_edge(START,"retrieve_relevant_docs_node")
builder.add_edge("retrieve_relevant_docs_node","grade_retrieved_documents")
builder.add_conditional_edges(
    "grade_retrieved_documents",
    decide_next_step,
    {"web_search": "web_search", "generate_node": "generate_node"},
)
builder.add_edge("web_search","generate_node")
builder.add_edge("generate_node",END)

graph = builder.compile()