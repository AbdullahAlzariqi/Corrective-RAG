from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.chains import answer_grader, hallucination_grader
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO THE QUERY---")
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state:GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.generation_hallucination_grader.invoke(
        {"generation":generation,"documents":documents}
    )
    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.answer_grader.invoke({"question":question, "generation":generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS THE QUESTION")
            return "not useful"
    else:
        print("DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, REVISE")
        return "not supported"

workflow = StateGraph(GraphState)

workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GENERATE, generate) 
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
   GRADE_DOCUMENTS,
   decide_to_generate)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={
        "not supported": GENERATE,
        "useful": END,
        "not useful":WEBSEARCH
    }
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app= workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")