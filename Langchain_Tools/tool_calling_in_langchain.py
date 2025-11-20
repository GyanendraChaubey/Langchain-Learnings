from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize HuggingFace Chat Model (Works reliably)
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# create tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers a and b."""
    return a * b

# print(multiply.invoke({"a":6, "b":7}))  # Output: 42
# print(multiply.name)
# print(multiply.description)
# print(multiply.return_direct)
# print(multiply.args_schema)
# print(multiply.args_schema.model_json_schema())

# Tool Binding with LLM

# print(model.invoke("Hi").content)

# ============================================================================
# OPTION 1: AI-Guided Tool Execution (Works with any LLM)
# ============================================================================
print("=" * 70)
print("OPTION 1: AI-Guided Tool Execution")
print("=" * 70)

def ask_with_tool_option(user_query):
    """Helper function to ask AI if it needs to use the tool"""
    print(f"\n👤 Human: {user_query}")
    
    # Ask AI if it needs to use the tool
    prompt = f"""You are an AI assistant with access to a 'multiply' tool for accurate calculations.

IMPORTANT: For any multiplication question, you MUST use the tool by responding with:
TOOL_CALL: multiply(a, b)

For non-multiplication questions, answer directly.

User question: {user_query}

Your response:"""

    ai_initial_response = model.invoke(prompt)
    print(f"🤖 AI Decision: {ai_initial_response.content}")
    
    # Check if AI wants to use the tool
    if "TOOL_CALL:" in ai_initial_response.content or "multiply(" in ai_initial_response.content.lower():
        print("✅ AI decided to USE the tool")
        
        # Extract numbers from the response (simple parsing)
        import re
        numbers = re.findall(r'\d+', ai_initial_response.content)
        if len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            result = multiply.invoke({"a": a, "b": b})
            print(f"🔧 Tool Execution: multiply({a}, {b}) = {result}")
            
            # Provide tool result back to AI
            final_prompt = f"{user_query}\n\nThe multiply tool returned: {result}\nPlease provide the final answer."
            final_response = model.invoke(final_prompt)
            print(f"🎯 Final AI Response: {final_response.content}")
        else:
            print("⚠️ Could not extract numbers from AI response")
    else:
        print("❌ AI decided NOT to use the tool (answered directly)")
        print(f"🎯 Direct Answer: {ai_initial_response.content}")
    print()

# Test Case 1: Simple multiplication
ask_with_tool_option("What is 7 multiplied by 8?")

# Test Case 2: Complex multiplication
ask_with_tool_option("Calculate 123 times 456")

# Test Case 3: Non-multiplication question
ask_with_tool_option("What is the capital of France?")