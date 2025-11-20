from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader


loader = PyMuPDFLoader("Langchain_Text_Splitters/YogaforMentalHealth.pdf")
documents = loader.lazy_load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=10,
    separator="",

)

result = text_splitter.split_documents(documents)


print(f"Number of documents: {len(result)}")
print(f"First document: {result[20].page_content}")
print(f"Second document: {result[21].page_content}")