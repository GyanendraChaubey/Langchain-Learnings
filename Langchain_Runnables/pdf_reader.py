from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader

load_dotenv()

# Load the document
loader = PyMuPDFLoader("Langchain_Runnables/YogaforMentalHealth.pdf")
documents = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# convert text into embeddings and store in vector database
vectorstore = FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

# Create a retriever fetches relevant documents
retriever = vectorstore.as_retriever()

# Manually retrieve relevant documents
age_input = "adult"
gender_input = "female"
stress_level_input = "high"
anxiety_level_input = "high"
focus_difficulty_input = "severe"
relaxation_level_input = "poor"
sleep_quality_input = "very poor"
emotional_flatness_input = "none"

query = "What is the suitable therapy for a {age_input} year old {gender_input} with {stress_level_input} stress, {anxiety_level_input} anxiety, {focus_difficulty_input} focus difficulty, {relaxation_level_input} relaxation level, {sleep_quality_input} sleep quality, and {emotional_flatness_input} emotional flatness?".format(
    age_input=age_input,
    gender_input=gender_input,
    stress_level_input=stress_level_input,
    anxiety_level_input=anxiety_level_input,
    focus_difficulty_input=focus_difficulty_input,
    relaxation_level_input=relaxation_level_input,
    sleep_quality_input=sleep_quality_input,
    emotional_flatness_input=emotional_flatness_input
)
retrieved_docs = retriever.get_relevant_documents(query)


# combine the retrieved documents into a single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])


# Initialize LLM endpoints
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Manually pass the retrieved text to the model
prompt = f"""Act as an alternative therapist.
Based on the following information from the PDF documents:
{retrieved_text}, 
can you suggest an appropriate therapy approach for a {age_input} year old {gender_input} with {stress_level_input} stress, {anxiety_level_input} anxiety, {focus_difficulty_input} focus difficulty, {relaxation_level_input} relaxation level, {sleep_quality_input} sleep quality, and {emotional_flatness_input} emotional flatness?
"""
response = model.predict(prompt)

# Print the model's response
print(response)