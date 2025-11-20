from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chat_template = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful therapy assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', "{query}")
])

chat_history = []


with open("Langchain_Prompts/chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())


print(chat_history)

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Can you provide some tips for managing anxiety?"
})

print(prompt)