from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


chat_prompt = ChatPromptTemplate(
    messages=[
            ('system', "You are a helpful {domain} assistant."),
            ('human', "Explain in simple terms, about {topic_name}."),
        # SystemMessage(content="You are a helpful {domain} assistant."),
        # HumanMessage(content="Explain in simple terms, about {topic_name}"),
    ]
)

prompt = chat_prompt.invoke({"domain": "therapy", "topic_name": "anxiety and stress"})

print(prompt)
