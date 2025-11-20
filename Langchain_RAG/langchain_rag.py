from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

## 1.a Ingestion Youtube video transcript
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # Get the transcript using the YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id)

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

## 1.b Indexing (Document Ingestion)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# print(chunks)
# print(chunks[10])

## 1.c Embedding generation and storing in vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)


# print(vectorstore.index_to_docstore_id[10])

# print(vectorstore.get_by_ids([vectorstore.index_to_docstore_id[10]]))

## 2 Retrieval

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# print(retriever)

# print(retriever.invoke('What is deepmind'))

## 3. Augmentation

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
#print("=*================ Retrieved Docs =*================")
#print(retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
print("=*================ Retrieved Context =*================")
print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})
print("=*================ Final Prompt =*================")
print(final_prompt)

## 4. Generate the final answer
answer = model.invoke(final_prompt)
print("=*================ Final Answer =*================")
print(answer.content)


