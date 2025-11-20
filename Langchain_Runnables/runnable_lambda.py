from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

def word_count(text: str) -> int:
    return len(text.split())

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

joke_chain = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)

})

final_chain = RunnableSequence(
    joke_chain,
    parallel_chain
)

result = final_chain.invoke({"topic": "Machine Learning"})

print(result)

print(final_chain.get_graph().draw_ascii())