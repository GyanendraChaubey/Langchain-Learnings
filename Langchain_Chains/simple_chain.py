from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template = PromptTemplate(template=
                          """Provide 3 interesting facts about the topic: {topic}.""", 
                          input_variables=["topic"])

chains = template | model | parser

result = chains.invoke(input="Generative AI")

print(result)