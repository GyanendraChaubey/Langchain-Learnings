from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.header("Research Tool")


paper_input = st.selectbox( "Select Research Paper Name",
    ["Select Paper.....", "Attention Is All You Need",
      "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 
      "GPT-3: Language Models are Few-Shot Learners", 
      "T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
      "Diffusion models in Vision: A Survey"]
)

style_input = st.selectbox("Select Explanation Style",
    ["Select Style.....", "Simple Explanation", "Detailed Explanation", "Technical Explanation", "Analogy-based Explanation"]
)

length_input = st.selectbox("Select Explanation Length",
    ["Select Length.....", "Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (5+ paragraphs)"]
)

template = load_prompt("Langchain_Prompts/prompt.json")

# prompt = template.invoke({"paper_input": paper_input, "style_input": style_input, "length_input": length_input})

if st.button("Search"):
    chain = template | model
    result = chain.invoke(
        {"paper_input": paper_input,
         "style_input": style_input, 
         "length_input": length_input})
    st.text(result.content)
