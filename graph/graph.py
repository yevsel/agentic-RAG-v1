from langgraph.graph import StateGraph, START, END
from graph.nodes.retrieve import retrieve_relevant_docs_node
from graph.nodes.grade_documents import grade_retrieved_documents
from graph.nodes.web_search import web_search
from graph.nodes.generate import generate_node
from graph.state import GraphState
from graph.nodes.edges import decide_next_step, decide_node_after_grading_documents_and_llm_answer_groundness, route_question




builder = StateGraph(GraphState)

builder.add_node("retrieve_relevant_docs_node",retrieve_relevant_docs_node)
builder.add_node("grade_retrieved_documents",grade_retrieved_documents)
builder.add_node("web_search",web_search)
builder.add_node("generate_node",generate_node)

builder.add_conditional_edges(
    START,
    route_question,
    {"web_search": "web_search", "retrieve_relevant_docs_node": "retrieve_relevant_docs_node"},
)
builder.add_edge("retrieve_relevant_docs_node","grade_retrieved_documents")
builder.add_conditional_edges(
    "grade_retrieved_documents",
    decide_next_step,
    {"web_search": "web_search", "generate_node": "generate_node"},
)
builder.add_edge("web_search","generate_node")
builder.add_conditional_edges(
    "generate_node",
    decide_node_after_grading_documents_and_llm_answer_groundness,
    {
        "not_supported": "generate_node",
        "not_useful": "web_search",
        "useful": END,
    },
)

graph = builder.compile()