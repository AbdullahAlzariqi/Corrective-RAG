from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """
You are a Large Language Model tasked with evaluating the relevance of a document to a specific query. You will be provided with the full text of a document and a query. Your task is to determine if the document is relevant to the query.

Instructions:

Analyze the Query: Carefully read and understand the user's query. Identify the key concepts, entities, and the user's intent.
Analyze the Document: Thoroughly read the document, paying attention to its main points, supporting arguments, and any specific details that might be relevant to the query.
Assess Relevance: Determine if the document provides information that directly addresses the user's query. Consider the following factors:
Topical Relevance: Does the document cover the same topic as the query?
Informativeness: Does the document provide information that would be useful to someone seeking an answer to the query?
Specificity: Does the document address the specific aspects or nuances of the query?
Perspective: Does the document offer a perspective or opinion relevant to the query, even if it doesn't directly answer it?
Handle Ambiguity: If the query is ambiguous or open to interpretation, consider multiple possible interpretations and assess relevance based on the most likely interpretations.
Respond with "Yes" or "No": Based on your analysis, provide a definitive answer:
"Yes": if the document is relevant to the query.
"No": if the document is not relevant to the query.
Important Considerations:

Avoid Assumptions: Do not make assumptions about the user's intent or knowledge. Base your judgment solely on the information provided in the query and the document.
Consider Different Types of Relevance: Relevance can take many forms. A document may be relevant even if it doesn't directly answer the query. For example, it could provide background information, context, or alternative perspectives.
Be Thorough: Carefully consider all aspects of the document and the query before making your judgment.
Example:

Query: "What are the health benefits of eating apples?"

Document: "Apples are a popular fruit, enjoyed for their sweet taste and versatility in cooking. They are a good source of fiber and vitamin C."

Answer: Yes

Explanation: While the document doesn't explicitly list the health benefits of apples, it does mention that they are a good source of fiber and vitamin C, which are relevant to the query about health benefits.

Remember: Your primary goal is to accurately assess the relevance of the document to the query. Be objective, thorough, and focus on providing a clear and concise answer.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User Question: {question}")

    ]
)

retrieval_grader = grade_prompt | structured_llm_grader 
