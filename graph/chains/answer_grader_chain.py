from graph.llm_model import llm
from graph.objects import AnswerGraderObject
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
structured_llm_grader = llm.with_structured_output(AnswerGraderObject)

answer_grader_chain_system_message="""
    You are a grader assessing whether an answer addresses / resolves a question \n
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question
"""

answer_grader_chain_prompt_template=ChatPromptTemplate(
    [
        ("system",answer_grader_chain_system_message),
        ("human","Question: \n\n {question} \n\n LLM answer: {generation}")
    ]
)

answer_grader_chain = answer_grader_chain_prompt_template | structured_llm_grader