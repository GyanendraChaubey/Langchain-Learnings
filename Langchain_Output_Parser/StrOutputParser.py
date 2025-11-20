from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

prompt1 = template1.invoke(input="India")

result1 = model.invoke(prompt1)

print("Detailed Report:\n", result1.content)

# 2nd Prompt: concise Summary
template2 = PromptTemplate(
    template="Summarize the 5 key points about {text}",
    input_variables=["text"]
)

prompt2 = template2.invoke(input=result1.content)

result2 = model.invoke(prompt2)

print("Concise Summary:\n", result2.content)