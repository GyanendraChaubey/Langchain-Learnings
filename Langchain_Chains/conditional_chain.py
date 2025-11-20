from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback as positive or negative.")

parser = StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template= """Generate the sentiment in positive or negative based on the given feedback: {feedback}. \n {format_instructions}""",
    input_variables=["feedback"],
    partial_variables={ "format_instructions": parser2.get_format_instructions() }
)

classifier_chain = prompt1 | model | parser2

# feedback = "The product quality is bad and delivery was prompt."
# result = classifier_chain.invoke(input=feedback)
# print(result)

prompt2 = PromptTemplate(
    template= "Write an appropriate response to the following positive feedback: {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template= "Write an appropriate response to the following negative feedback: {feedback}",
    input_variables=["feedback"]
)


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive',
    prompt2 | model | parser
    ),
    (lambda x: x.sentiment == 'negative',
    prompt3 | model | parser
    ),
    RunnableLambda(lambda x: "No sentiment detected.")
)

chain = classifier_chain | branch_chain

feedback = "The product is wonderful."
result = chain.invoke(input=feedback)
print(result)

print(chain.get_graph().print_ascii())