from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the fictional person")
    age: int = Field(gt=18,description="The age of the fictional person")
    city: str = Field(description="The city where the fictional person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a {place} person.\n {format_instructions} ",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.invoke(input="Indian")

# response = model.invoke(prompt)

# parsed_output = parser.parse(response.content)

# print(parsed_output)

chain = template | model | parser

result = chain.invoke(input="Indian")
print(result)