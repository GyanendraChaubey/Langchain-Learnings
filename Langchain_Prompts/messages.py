from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = [
    SystemMessage(content="You are a helpful Psycho Therapy Assistant."),
    HumanMessage(content="I am feeling very anxious and stressed lately. Can you help me?"),
]

response = model.invoke(messages)

messages.append(AIMessage(content=response.content))

print("Therapy Assistant:", response.content)

print(messages)
