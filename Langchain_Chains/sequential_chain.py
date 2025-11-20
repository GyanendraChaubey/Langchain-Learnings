from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1 = PromptTemplate(
    template="""Give a detailed report on topic {topic}""",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="""Summarize the report on topic {text} in 5 key points.""",
    input_variables=["text"],
)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke(input="Generative AI")

print(result)

print(chain.get_graph().print_ascii())
