from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="tiiuae/Falcon3-1B-Instruct",
    task="text-generation",
    model_kwargs={"temperature": 0.7},
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(template=
                          """Give me the name, age and city of a fictional person. \n {format_instructions} """, 
                          input_variables=[], partial_variables={"format_instructions": parser.get_format_instructions()})

# prompt = template.invoke({})
# response = model.invoke(prompt)
# parsed_output = parser.parse(response.content)
# print(parsed_output)

chains = template | model | parser

result = chains.invoke({})

print(result)


## Biggest flaw of json output parser is that you can enforce json format but you cannot enforce schema validation like pydantic model.