from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

loader = DirectoryLoader("Langchain_Document_Loaders/papers/", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
# docs = loader.load()

# print(docs[12].metadata)
# print(docs[12].page_content)
# print(len(docs))

# Load vs Lazy Load
# Load put all the documents into memory at once while Lazy Load only loads documents when they are accessed.
# Lazy loader is more memory efficient for large document collections and fast as well as it only loads the required documents when needed.

lazy_docs = loader.lazy_load()

for i, doc in enumerate(lazy_docs):
    if i == 12:
        print(doc.metadata)
        print(doc.page_content)
        break