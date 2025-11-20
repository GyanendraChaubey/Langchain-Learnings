# Langchain Expressions Language (LCEL) file

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "Write a joke on the topic: {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

chain = template1 | model | parser # LCEL syntax for RunnableSequence

result = chain.invoke({"topic": "Machine Learning"})

print(result)