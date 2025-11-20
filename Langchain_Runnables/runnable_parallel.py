from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "Write a Twitter Post on the topic: {topic}",
    input_variables = ["topic"]
)


template2 = PromptTemplate(
    template= "Write a LinkedIn Post on the topic: {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = RunnableParallel(
    {"Twitter Post": template1 | model | parser,
     "LinkedIn Post": template2 | model | parser}
)

result = chain.invoke({"topic": "Artificial Intelligence in Healthcare"})

print(result)

print(chain.get_graph().draw_ascii())