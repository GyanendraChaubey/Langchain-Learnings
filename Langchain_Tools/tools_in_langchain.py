# Using Builtin Tools in Langchain

# --------------------------------------------------------------------------------
# Builtin Tools - DuckDuckGo Search Tool
#---------------------------------------------------------------------------------

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.run("Who is Gyanendra Chaubey from IIT Jodhpur?")

# print(results)

# print(search_tool.name)
# print(search_tool.description)
# print(search_tool.return_direct)
# print(search_tool.args_schema)

# --------------------------------------------------------------------------------
# Builtin Tools - Shell Command Tool
#---------------------------------------------------------------------------------

from langchain_community.tools import ShellTool
shell_tool = ShellTool()
output = shell_tool.run("echo Hello, Gyanendra!")
# print(output)

# print(shell_tool.name)
# print(shell_tool.description)
# print(shell_tool.return_direct)
# print(shell_tool.args_schema)

# --------------------------------------------------------------------------------
# Custom Tools - Method 1: using the @tool Decorator
#---------------------------------------------------------------------------------

from langchain_core.tools import tool

# Step 1: Define the function
def multiply(a, b):
    return a * b

# Step 2: Add with Hinting
def multiply(a: int, b: int) -> int:
    return a * b

# Step 3: Use the @tool Decorator
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers a and b."""
    return a * b

# Step 4: Using the Custom Tool
result = multiply.invoke({"a":6, "b":7})
# print(result)  # Output: 42

# print(multiply.name)
# print(multiply.description)
# print(multiply.return_direct)
# print(multiply.args_schema)
# print(multiply.args_schema.model_json_schema())


# --------------------------------------------------------------------------------
# Custom Tools - Method 2: using StructuredTool
#---------------------------------------------------------------------------------

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

def multiply_func(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiplies two numbers a and b.",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({"a":8, "b":9})
# print(result)  # Output: 72
# print(multiply_tool.name)
# print(multiply_tool.description)
# print(multiply_tool.return_direct)
# print(multiply_tool.args_schema)
# print(multiply_tool.args_schema.model_json_schema())

# --------------------------------------------------------------------------------
# Custom Tools - Method 3: Subclassing BaseTool
#---------------------------------------------------------------------------------

from langchain_core.tools import BaseTool
from typing import Type

class MultiplyInputModel(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiplies two numbers a and b."
    args_schema: Type[BaseModel] = MultiplyInputModel

    def _run(self, a: int, b: int) -> int:
        return a * b

multiply_tool = MultiplyTool()
result = multiply_tool.invoke({"a":10, "b":11})
# print(result)  # Output: 110
# print(multiply_tool.name)
# print(multiply_tool.description)
# print(multiply_tool.return_direct)
# print(multiply_tool.args_schema)
# print(multiply_tool.args_schema.model_json_schema())

# --------------------------------------------------------------------------------
# Toolkit - Combining Multiple Tools
#---------------------------------------------------------------------------------

from langchain_core.tools import Tool

# Create individual tools
@tool
def add(a: int, b: int) -> int:
    """Adds two numbers a and b."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts b from a."""
    return a - b

class MathToolkit():
    def get_tools(self) -> list[Tool]:
        return [add, subtract]

# Using the Toolkit
math_toolkit = MathToolkit()

tools = math_toolkit.get_tools()
add_result = tools[0].invoke({"a":5, "b":3})  # Using add tool
subtract_result = tools[1].invoke({"a":5, "b":3})  # Using subtract tool   

print(f"Addition Result: {add_result}")        # Output: 8
print(f"Subtraction Result: {subtract_result}")  # Output: 2

