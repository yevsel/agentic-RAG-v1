from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()
from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    user_question = state['question']
    retrieved_documents = state.get('retrieved_documents') or []

    tavily_response = web_search_tool.invoke({"query": user_question})

    web_content = "\n\n".join(result["content"] for result in tavily_response["results"])
    web_document = Document(page_content=web_content)

    return {"retrieved_documents": [web_document]}


if __name__ == "__main__":
    print(web_search({"question": "agent memory", "retrieved_documents": None}))
