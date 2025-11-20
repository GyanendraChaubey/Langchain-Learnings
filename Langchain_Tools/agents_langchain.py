from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.messages import HumanMessage, ToolMessage
from langchain import hub
import requests
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

  response = requests.get(url)

  return response.json()


search_tool = DuckDuckGoSearchRun()


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it"
)

model = ChatHuggingFace(llm=llm, verbose=True)


# Step 2: Pull the React Prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")


# Step 3: Create the ReAct Agent Manually
react_agent = create_react_agent(
    llm=model,
    tools=[get_weather_data, search_tool],
    prompt=prompt
)

# Step 4: Create the Agent Executor
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=[get_weather_data, search_tool],
    verbose=True
)

# --- Step 5: Run the Agent ---
user_query = "What is the weather in New York City?"

response = agent_executor.invoke({"input": user_query})

print(f"\nFinal Response: {response['output']}")





