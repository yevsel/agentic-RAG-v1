from typing import Any, Dict

from graph.state import GraphState
from graph.chains.generation import generation_chain


def generate_node(state: GraphState) -> Dict[str, Any]:
    user_question = state['question']
    retrieved_documents = state['retrieved_documents']

    context = "\n\n".join(doc.page_content for doc in retrieved_documents)
    answer = generation_chain.invoke({"question": user_question, "context": context})

    return {"generation": answer}
