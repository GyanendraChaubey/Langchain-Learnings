from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
)
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description="First fact about the topic", type="string"),
    ResponseSchema(name="fact_2", description="Second fact about the topic", type="string"),
    ResponseSchema(name="fact_3", description="Third fact about the topic", type="string"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(template=
                          """Provide 3 interesting facts about the topic: {topic}.\n {format_instructions} """, 
                          input_variables=["topic"], partial_variables={"format_instructions": parser.get_format_instructions()})

chain = template | model | parser

result = chain.invoke(input="Generative AI")

print(result)


# This can enforce the model to respond in a structured format based on the defined schema but can not enforce correct data types or values (data validation).

