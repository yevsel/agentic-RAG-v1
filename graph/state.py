from typing import List, TypedDict


class GraphState(TypedDict):
    question: str # storing the question asked
    generation: str # This is our LLM generated answer
    web_search_flag: bool # Boolean to tell us whether to search online for extra information
    retrieved_documents: List[str] # Retrieved documents from our VectorDB
    
    