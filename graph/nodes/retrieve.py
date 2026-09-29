from graph.state import GraphState
from typing import Dict, Any
from ingestion import chroma_docs_retriever


def retrieve_relevant_docs_node(state: GraphState):
    user_question = state['question']
    # Get the relevant docs from Chroma
    relevant_docs = chroma_docs_retriever.invoke(user_question)
    
    return {"retrieved_documents": relevant_docs}
    
    