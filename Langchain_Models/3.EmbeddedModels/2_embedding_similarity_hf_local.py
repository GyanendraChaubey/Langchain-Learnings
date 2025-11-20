from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = ["My name is Gyanendra.", "Gyanendra live in India.", "Gyanendra is a Computer Scientist."]

query = "Tell me about Gyanendra?"

docs_embedd = embeddings.embed_documents(docs)
query_embedd = embeddings.embed_query(query)

scores = similarities = cosine_similarity([query_embedd], docs_embedd)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(docs[index])
print(f"Similarity Score: {score}")