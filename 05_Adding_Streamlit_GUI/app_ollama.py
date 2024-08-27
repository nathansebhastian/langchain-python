from langchain_ollama.chat_models import ChatOllama
import streamlit as st

llm = ChatOllama(model="gemma:2b")

st.title("Q & A With AI")

question = st.text_input("Your Question")

if question:
    response = llm.invoke(question)
    st.write(response.content)