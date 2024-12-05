from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

urls = [
    "https://lilianweng.github.io/posts/2020-10-29-odqa/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/"
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist ]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="advanced_Rag",
#     embedding=embeddings,
#     persist_directory="./.chroma"
#     )

retriever = Chroma(
    collection_name="advanced_Rag",
    embedding_function=embeddings,
    persist_directory="./.chroma"
).as_retriever()