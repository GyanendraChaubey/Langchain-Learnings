from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Step1: Your source documents
source_docs = [
    Document(page_content="Yoga therapy can significantly improve mental health by reducing stress, anxiety, and depression."),
    Document(page_content="Regular practice of yoga therapy enhances emotional well-being and promotes relaxation."),
    Document(page_content="Yoga therapy combines physical postures, breathing exercises, and meditation to support mental health."),
]

# Step2: Initialize model embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Step3: Create a Chroma vector store from the source documents
vectorstore = Chroma.from_documents(
    documents=source_docs,
    embedding=embeddings,
    collection_name="yoga_therapy_docs",
)

# Step4: Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Step5: Use the retriever to fetch relevant documents based on a query
query = "How does yoga therapy help with anxiety and stress?"
relevant_docs = retriever.invoke(query)

print("Retrieved Relevant Documents:")
for doc in relevant_docs:
    print(f"- {doc.page_content}")  # Print the full content of each retrieved document 


