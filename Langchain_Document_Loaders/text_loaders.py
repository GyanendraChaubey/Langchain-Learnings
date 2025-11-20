from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader



load_dotenv()

loader = TextLoader("Langchain_Document_Loaders/poem.txt")
documents = loader.load()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = "Generate a poem on the topic: {poem}",
    input_variables = ["poem"]
)

parser = StrOutputParser()

print(documents)
print(documents[0].page_content)
print(documents[0].metadata)
print(len(documents))
print(type(documents))



chain = prompt | model | parser

result = chain.invoke({"poem": documents[0].page_content})

print("Poem Summary:", result)

