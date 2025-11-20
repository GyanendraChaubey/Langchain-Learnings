from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


url = "https://scholar.google.com/citations?user=UKNVEgQAAAAJ&hl=en"

loader = WebBaseLoader(url)

data = loader.load()

print(f"Number of documents loaded: {len(data)}")
# print(f"Content of the first document:\n{data[0].page_content}")

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    input_variables=["data", "question"],
    template="Answer the following {question} from the following content:\n{data}",
)

parser = StrOutputParser()

question = input("Enter your question: ")

chain = prompt | model | parser

response = chain.invoke({
    "data": data[0].page_content,
    "question": question
})

print("Response:", response)


# Make a chome plugin which can chat to any page you are visiting.
# Use langchain webbase loader to load the page content and then use the llm to answer questions about the page.
# You can use selenium to get the current page url and then use the webbase loader to load the page content.
# You can use streamlit to create the chat interface.
# You can use streamlit's session state to store the chat history.
# You can use streamlit's text input to get the user's question.
# You can use streamlit's button to submit the question.
# You can use streamlit's chat message to display the chat history.
# You can use streamlit's markdown to display the response from the llm.

