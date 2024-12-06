from typing import List
from langchain.schema.runnable import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

class GradeHallucination(BaseModel):
    binary_score: str = Field(
        description="retrieved set of facts are grounding / supporting the answer, 'yes' or 'no'"
    )
    hallucinations: List[str] = Field(
        description="A list of strings that are hallucinations. Hallucinations are answers that are not relevant to the question"
    )


system = """ You are a grader adressing wehter an LLM generation is grounded in /  supported by a set of retrieved facts. 
Guve a binary score of 'yes' or 'no'. 'yes' measn that the answer is grounded in / sppurted by the retrieved facts
"""

structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucination)

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "llm generation: {generation} \n\n Set of Facts:\n\n {documents}")

    ]
)

generation_hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader