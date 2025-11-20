from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chatmodel = ChatAnthropic(model_name="claude-2")

result = chatmodel.invoke("What is the capital of India?")

print(result)

print(result.content)
