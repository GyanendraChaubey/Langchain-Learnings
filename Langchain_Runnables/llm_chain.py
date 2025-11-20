from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
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

# Define LLM Chain
llm_chain = LLMChain(llm=model, prompt=prompt)

# Define the input
input = input("Enter a blog post topic: ")

# Call the LLM Chain with the input
blog_title = llm_chain.run(input)

# Print the generated blog title
print(f"Generated Blog Title: {blog_title}")