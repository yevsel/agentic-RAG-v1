from graph.objects import IntentClassifierObject
from graph.llm_model import llm
from langchain_core.prompts import ChatPromptTemplate


structured_llm_output = llm.with_structured_output(IntentClassifierObject)

intent_classifier_system_prompt = """
    You are an expert at routing a user question to a vectorstore or a websearch
    The vectorstore contains documents related to agents, prompt engineering and adversiral attacks
    Use the vectorstore for questions on these topics. For anything else use web search.
"""


intent_classifier_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",intent_classifier_system_prompt),
        ("human","{question}")
    ]
)

intent_classifier_chain = intent_classifier_prompt_template | structured_llm_output