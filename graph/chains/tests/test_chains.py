from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import GradeDocumentsObject, retrieval_grader_chain
from ingestion import chroma_docs_retriever


def test_retrieval_grader_answer_yes() -> None:
    user_question = "agent-memory"
    docs = chroma_docs_retriever.invoke(user_question)
    docs_txt = docs[1].page_content
    
    response: GradeDocumentsObject =  retrieval_grader_chain.invoke({
        "document": docs_txt,
        "question": user_question
    })
    
    assert response.binary_score == "yes"
    
    
def test_retrieval_grader_answer_no() -> None:
    user_question = "agent-memory"
    docs = chroma_docs_retriever.invoke(user_question)
    docs_txt = docs[1].page_content
    
    response: GradeDocumentsObject =  retrieval_grader_chain.invoke({
        "document": docs_txt,
        "question": "How to make pizza"
    })
    
    assert response.binary_score == "no"