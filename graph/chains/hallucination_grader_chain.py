from graph.llm_model import llm
from graph.objects import HallucinationGraderObject
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
structured_llm_grader = llm.with_structured_output(HallucinationGraderObject)

hallucination_grader_chain_system_message="""
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved documents
    Give a binary score of 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

hallucination_grader_chain_prompt_template=ChatPromptTemplate(
    [
        ("system",hallucination_grader_chain_system_message),
        ("human","Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)

hallucination_grader_chain = hallucination_grader_chain_prompt_template | structured_llm_grader