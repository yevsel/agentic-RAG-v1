from dotenv import load_dotenv

from ..hallucination_grader_chain import hallucination_grader_chain
from ..generation import generation_chain



from graph.chains.retrieval_grader import GradeDocumentsObject, retrieval_grader_chain
from ingestion import chroma_docs_retriever
load_dotenv()

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
    
    
def test_hallucination_grader_answer_yes():
    question = "agent memory"
    docs = chroma_docs_retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate LLM answer
    llm_answer_generation = generation_chain.invoke({"question":question,"context":context})

    # Hallucination checking
    response = hallucination_grader_chain.invoke({"documents":context, "generation":llm_answer_generation})

    assert response.binary_score is True
    

def test_hallucination_grader_answer_no():
    question = "agent memory"
    docs = chroma_docs_retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate LLM answer
    llm_answer_generation = generation_chain.invoke({"question":question,"context":context})

    # Hallucination checking
    response = hallucination_grader_chain.invoke({"documents":context, "generation":"Sausage pizza is delicious"})

    assert response.binary_score is False