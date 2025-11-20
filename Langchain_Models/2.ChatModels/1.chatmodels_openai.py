from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chatmodel = ChatOpenAI(model_name="gpt-3.5-turbo")

result = chatmodel.invoke("What is the capital of India?")

print(result)

print(result.content)