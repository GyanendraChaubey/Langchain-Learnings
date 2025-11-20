from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


chatmodel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = chatmodel.invoke("Who is Gyanendra Chaubey of IIT Jodhpur?")

print(result.content)
