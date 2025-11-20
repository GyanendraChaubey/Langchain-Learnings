from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


loader = PyPDFLoader("Langchain_Document_Loaders/YogaforMentalHealth.pdf")
docs = loader.load()

print(docs[11].metadata)

