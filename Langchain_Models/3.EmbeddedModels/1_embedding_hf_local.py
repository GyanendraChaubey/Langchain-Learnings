from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = ["My name is Gyanendra.", "I live in India.", "I am a Computer Scientist."]

vectors = embeddings.embed_documents(docs)

print(vectors)

#vector = embeddings.embed_query("Hello world")
#print(vector)