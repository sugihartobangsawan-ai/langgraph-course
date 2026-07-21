from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,chunk_overlap=250
)

doc_splits = text_splitter.split_documents(docs_list)

print(f"Documents: {len(docs_list)}")
print(f"Chunks: {len(doc_splits)}")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

#ONLY CALL THIS FOR INITIALIZATION
# vectorstore = Chroma.from_documents(
#     documents = doc_splits,
#     collection_name='rag-chroma',
#     embedding=embeddings,
#     persist_directory='./.chroma'
# )

retriever = Chroma(
    collection_name='rag-chroma',
    persist_directory='./.chroma',
    embedding_function=embeddings,
).as_retriever()