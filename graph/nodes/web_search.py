from typing import Any, Dict
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from graph.state import GraphState

load_dotenv()
websearch_tool = TavilySearchResults(max_results=3)


def web_search(state:GraphState) -> Dict[str,Any]:
    print("--WEB SEARCH--")
    question = state["question"]
    documents = state["documents"]

    tavily_results = websearch_tool.invoke({"query":question})
    joined_tavily_result = "\n".join([tavily_result["content"] for tavily_result in tavily_results])
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    return {"documents":documents, "question":question}