from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


# Use HuggingFace Inference API with a chat-optimized model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=10,
    temperature=0.7,
)

chatmodel = ChatHuggingFace(llm=llm)

result = chatmodel.invoke("What is the capital of India?")
print(result.content)
