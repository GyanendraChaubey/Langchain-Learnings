from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = [
    SystemMessage(content="You are a helpful therapy assistant. Provide empathetic and supportive responses to users seeking mental health support."),
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break

    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Therapy Assistant:", response.content)

print(chat_history)

# Save chat history to a text file
with open("chat_history.txt", "w") as f:
    for message in chat_history:
        f.write(f"{message.__class__.__name__}: {message.content}\n")


