from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


text = """
There was a farmer working in his field when he noticed a young boy.
The grass was green of the grass and the sky was blue. The cricket is 
a interestting match to watch. I love to watch crciket and play it too. 
I have played several matches in my school and college days.

The terriorsim is a big threat to the world. Many countries are working together
to fight against the terriorism. It is important to stay vigilant and report any
suspicious activities to the authorities.
"""

# Create a Document object directly from the text
documents = [Document(page_content=text)]

# Initialize SemanticChunker with Hugging Face embedding model
text_splitter = SemanticChunker(
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=75
)

result = text_splitter.split_documents(documents)

print(f"Number of documents: {len(result)}")
# print(f"First document: {result[0].page_content}")
# print(f"Second document: {result[1].page_content}")
print(result)

