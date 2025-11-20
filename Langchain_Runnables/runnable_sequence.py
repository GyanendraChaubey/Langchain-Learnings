from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


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

template2 = PromptTemplate(
    template = "Explain the following joke in brief: {joke}",
    input_variables = ["joke"]
)

parser = StrOutputParser()

chain = RunnableSequence(template1, model, parser, RunnableLambda(lambda x: {"joke": x}), template2, model, parser)

result = chain.invoke({"topic": "Machine Learning"})

print(result)

print(chain.get_graph().draw_ascii())