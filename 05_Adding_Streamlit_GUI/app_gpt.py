from decouple import config
from langchain_openai import ChatOpenAI
import streamlit as st

OPENAI_KEY = config("OPENAI_KEY")

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_KEY)

st.title("Q & A With AI")

question = st.text_input("Your Question")

if question:
    response = llm.invoke(question)
    st.write(response.content)