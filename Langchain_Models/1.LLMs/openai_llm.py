from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

print(llm("What is the capital of India?"))