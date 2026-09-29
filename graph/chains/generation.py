from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from graph.llm_model import llm

generation_prompt="""
    You are an assistant for a question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say you don't know.
    Use three sentences maximum and keep the answer concise \n
    Question: {question} \n Context: {context} \n Answer:
"""

generation_chat_template = ChatPromptTemplate.from_template(generation_prompt)

generation_chain = generation_chat_template | llm | StrOutputParser()