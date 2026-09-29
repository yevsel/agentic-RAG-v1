from langchain_core.prompts import ChatPromptTemplate
from graph.llm_model import llm
from graph.objects import GradeDocumentsObject
from graph.prompts import grader_system_prompt
from langchain_core.messages import SystemMessage, HumanMessage

structured_llm_grader_output = llm.with_structured_output(GradeDocumentsObject)


grader_chat_template = ChatPromptTemplate(
    [
        ("system", grader_system_prompt),
        ("human","Retrieved document: \n\n {document} \n\n User question: {question}")
    ]
)

retrieval_grader_chain = grader_chat_template | structured_llm_grader_output