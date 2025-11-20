from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM endpoints
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Create a prompt template
prompt = PromptTemplate(
    template="Suggest a catchy title for a blog post about: {topic}",
    input_variables=["topic"]
)

# Define the input
input = input("Enter a blog post topic: ")

# Format the prompt manually using prompt template
formatted_prompt = prompt.format(topic=input)

# Call the LLM with the formatted prompt
blog_title = model.predict(formatted_prompt)

# Print the generated blog title
print(f"Generated Blog Title: {blog_title}")