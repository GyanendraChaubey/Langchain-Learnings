from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

load_dotenv()


embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


loader = PyMuPDFLoader("Lanchain_Vector_Store/YogaforMentalHealth.pdf")
documents = loader.load()

print(len(documents))

vectorstore = Chroma(
    embedding_function=embed,
    collection_name="yoga_therapy_docs",
    persist_directory="Lanchain_Vector_Store/chroma_db"
)

vectorstore.add_documents(documents)

# result = vectorstore.get(include=["metadatas", "documents", "embeddings"], limit=5)

# print(result)


# retrieved_docs = vectorstore.similarity_search(
#     """What is the suitable therapy for 
#         a 25 year old female with high stress, high anxiety, 
#         severe focus difficulty, poor relaxation level, very poor sleep quality, 
#         and none emotional flatness?""", k=1)

# for doc in retrieved_docs:
#     print(doc.page_content)
#     print(doc.metadata)
#     print("-----")

# retrieved_docs_with_scores = vectorstore.similarity_search_with_score(
#     """What is the suitable therapy for 
#         a 25 year old female with high stress, high anxiety,
#         severe focus difficulty, poor relaxation level, very poor sleep quality,
#         and none emotional flatness?""", k=1)

# for doc, score in retrieved_docs_with_scores:
#     print(doc.page_content)
#     print(doc.metadata)
#     print(f"Score: {score}")
#     print("-----")

# meta data filtering
# retrieved_docs_filtered = vectorstore.similarity_search_with_score(
#     query ="",
#     filter={"Therapy": "Yoga and Prayam"}, k=1)

# for doc, score in retrieved_docs_filtered:
#     print(doc.page_content)
#     print(doc.metadata)
#     print(f"Score: {score}")
#     print("-----")


# update documents
doc = vectorstore.get(limit=1, include=["documents", "metadatas"])
print("Before update:")
print(doc["documents"][0].split("\n"))
print(doc["metadatas"][0])
print(doc['ids'])

updated_doc = Document(
    page_content="Updated content for the document.",
    metadata={"Therapy": "Updated Therapy Info"}
)
vectorstore.update_document(document=updated_doc, document_id=doc['ids'][0])
print("After update:")
updated = vectorstore.get(limit=1, include=["documents", "metadatas"])
print(updated["documents"][0].split("\n"))
print(updated["metadatas"][0])
print(updated['ids'])

# delete documents
vectorstore.delete(ids=doc['ids'][0])
# view documents after deletion
deleted = vectorstore.get(limit=1, include=["documents", "metadatas"])
print("After deletion:")
print(deleted["documents"][0].split("\n"))
print(deleted["metadatas"][0])
print(deleted['ids'])

