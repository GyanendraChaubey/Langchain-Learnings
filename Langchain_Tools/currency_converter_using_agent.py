# tool create
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage, ToolMessage
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'

  response = requests.get(url)

  return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate


# print(convert.args)

# print(get_conversion_factor.args)

# print(get_conversion_factor.invoke({"base_currency":"USD", "target_currency":"INR"}))

# print(convert.invoke({"base_currency_value":100, "conversion_rate":82.5}))

# Using HuggingFacePipeline - runs model locally (supports bind_tools!)
llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it"
)

model = ChatHuggingFace(llm=llm, verbose=True)

print("✅ Model loaded successfully!\n")

# Bind tools to the model
# model_with_tools = model.bind_tools([get_conversion_factor, convert])


agent_executor = initialize_agent(
    tools=[get_conversion_factor, convert],
    llm=model,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # using ReAct pattern
    verbose=True  # shows internal thinking
)

# --- Step 6: Run the Agent ---
user_query = "What is the conversion of 125000 INR to USD?"

response = agent_executor.invoke({"input": user_query})