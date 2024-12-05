from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

prompt = hub.pull("rlm/rag-prompt")

generation_chain = (prompt | llm | StrOutputParser())