from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

def word_count(text: str) -> int:
    return len(text.split())

model = ChatHuggingFace(llm=llm)


template1 = PromptTemplate(
    template = "Write a report on the topic: {topic}",
)

parser = StrOutputParser()

word_count_chain = RunnableLambda(word_count)

# Generate the report first
report_chain = RunnableSequence(template1, model, parser)

template2 = PromptTemplate(
    template = "Summarize the following report in brief: {report}",
    input_variables = ["report"]
)

# Branch based on word count: if > 50 words, summarize; if <= 50 return as-is
branch_chain = RunnableBranch(
    (lambda x: word_count(x) > 50,  # Condition 1: if word count > 50
    RunnableSequence(
        RunnableLambda(lambda x: {"report": x}),  # Convert string to dict
        template2,
        model,
        parser
    )),
    RunnableLambda(lambda x: x)  # Default fallback (should not reach here)
)

chain = RunnableSequence(
    report_chain,
    branch_chain
)

result = chain.invoke({"topic": "Artificial Intelligence in Healthcare"})

print(result)

print(chain.get_graph().draw_ascii())
     


