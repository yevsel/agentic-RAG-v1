from graph.state import GraphState
from graph.chains.hallucination_grader_chain import hallucination_grader_chain
from graph.chains.answer_grader_chain import answer_grader_chain

def decide_next_step(state: GraphState):
    """If the grader rejected any document, go search the web; otherwise generate the answer."""
    if state["web_search_flag"]:
        return "web_search"
    return "generate_node"

def decide_node_after_grading_documents_and_llm_answer_groundness(state: GraphState):
    """
    Self-RAG style check on the generated answer:
    1. Is it grounded in the retrieved documents (not hallucinated)?
    2. If grounded, does it actually answer the question?
    """
    user_question = state["question"]
    retrieved_documents = state["retrieved_documents"]
    llm_generated_answer = state["generation"]

    context = "\n\n".join(doc.page_content for doc in retrieved_documents)

    groundness_score = hallucination_grader_chain.invoke(
        {"documents": context, "generation": llm_generated_answer}
    )
    
    # If the generated docs and the llm answer dont correspond
    if not groundness_score.binary_score:
        return "not_supported"

    # Checking if the question and the llm generated answer correspond
    answer_score = answer_grader_chain.invoke(
        {"question": user_question, "generation": llm_generated_answer}
    )

    if answer_score.binary_score:
        return "useful"
    return "not_useful"
