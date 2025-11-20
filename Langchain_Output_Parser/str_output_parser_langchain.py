from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="tiiuae/Falcon3-1B-Instruct",
    task="text-generation",
    model_kwargs={"temperature": 0.7},
)

model = ChatHuggingFace(llm=llm)

# 1st Prompt: detailed Report
template1 = PromptTemplate(
    template="Generate a report on the {topic}",
    input_variables=["topic"]
)

# 2nd Prompt: concise Summary
template2 = PromptTemplate(
    template="Summarize the 5 key points about {text}",
    input_variables=["text"]
)


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke(input="Generative AI")

print(result)