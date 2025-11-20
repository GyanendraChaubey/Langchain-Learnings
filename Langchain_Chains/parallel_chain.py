from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)

model2 = ChatHuggingFace(llm=llm2)


parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = """Generate simple and consie notes based on the given text: {text}""",
    input_variables=["text"]
)  

prompt2 = PromptTemplate(
    template = """ Generate a quiz question based on the following text: {text}""",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template = """ Combine the following notes and quiz question into a single document:
    Notes: {notes}
    Quiz Question: {question}
    """
)


parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "question": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """LangChain is a framework for developing applications powered by language models. It provides a standard interface for all language models, as well as tools to work with prompts, chains, agents, and memory. LangChain is designed to help developers build applications that can reason, plan, and interact with the world in a more human-like way."""
result = chain.invoke(input=text)

print(result)

print(chain.get_graph().print_ascii())

