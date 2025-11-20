"""
MMR (Maximal Marginal Relevance) Retriever

- It retrieves documents by balancing relevance and diversity, ensuring that 
    the selected documents are not only pertinent to the query but also provide 
    varied perspectives.

MMR is a algorithm designed to reduce redundancy while maintaining relevance in 
information retrieval tasks.

Why MMR Retriever?
- Balances Relevance and Diversity: MMR ensures that the retrieved documents are 
  not only relevant to the query but also diverse, reducing redundancy.
- Improved User Experience: By providing a variety of perspectives, MMR enhances
  the user experience, especially in applications like chatbots and 
  recommendation systems.
- Effective for Complex Queries: MMR is particularly useful for complex queries 
  where multiple facets need to be addressed.

In regular query search, you may get the docs that are:
- All similar to each other
- Repeating same information
- Lacking variety in perspectives

In MMR, the retriever selects documents that are:
- Relevant to the query
- Diverse in content
- Offering multiple viewpoints

This helps in RAG pipelines where:
- you want the context window to contain diverse but still relevant information
- Especially useful when documents are semantically overlapping
"""

from langchain_core.documents import Document


docs = [
    Document(page_content="Langchain makes it easy to work with LLMs"),
    Document(page_content="Langchain is used to build LLM based applications"),
    Document(page_content="Chroma is used to store and search document embeddings"),
    Document(page_content="Embeddings are vector representations of work"),
    Document(page_content="MMR helps you get diverse results when ")
]

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")  


# Create a FAISS vector store from documents
vectorstore = FAISS.from_documents(docs, embeddings)

# Enable MMR retriever
retriever = vectorstore.as_retriever(
    search="mmr",  # This enables MMR search
    search_kwargs={"k": 3, "lambda_mult": 1}) # top k documents, lambda for diversity vs relevance


query = "What is Langchain used for?"

retrieved_docs = retriever.invoke(query)


for i, doc in enumerate(retrieved_docs):
    print(f"Document {i+1}: {doc.page_content}")



