from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader_chain


def grade_retrieved_documents(state:GraphState):
    """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search
    """
    retrieved_documents= state['retrieved_documents']
    user_question=state['question']
    
    filtered_documents=[]
    web_search_flag=False
    for document in retrieved_documents:
        score = retrieval_grader_chain.invoke({"question":user_question, "document": document.page_content})
        grade = score.binary_score
        if grade.lower() == "yes":
            filtered_documents.append(document)
        else:
            web_search_flag = True
            continue
    return {"retrieved_documents": filtered_documents, "question":user_question, "web_search_flag":web_search_flag}
        