from langchain_community.retrievers import WikipediaRetriever

# Initialize the Wikipedia Retriever
retriever = WikipediaRetriever(
    top_k=1,  # Number of relevant articles to retrieve
    language="en"  # Language of the Wikipedia articles
)

query = "What are the benefits of yoga therapy for mental health?"

# Retrieve relevant Wikipedia articles
docs = retriever.invoke(query)

print("Retrieved Wikipedia Articles:")
for doc in docs:
    print(f"- {doc.page_content}")  # Print the full content of each article