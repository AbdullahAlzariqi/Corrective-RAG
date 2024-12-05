from typing import List, TypedDict



class GraphState(TypedDict):
    """
    Represents the state of our graph

    Attributes:
        question: question
        generation: LLM generation 
        web_search: wether toa dd web search
        documents: list of documents
    """

    question: str
    generation: str 
    web_search: bool
    documents: List[str]