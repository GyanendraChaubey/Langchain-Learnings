from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="tiiuae/Falcon3-1B-Instruct",
    task="text-generation",
    model_kwargs={"temperature": 0.7},
)

chatmodel = ChatHuggingFace(llm=llm)

result = chatmodel.invoke("What is the capital of India?")
print(result.content)
