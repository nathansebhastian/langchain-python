from langchain_ollama.chat_models import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.globals import set_debug

set_debug(True)

llm = ChatOllama(model="mistral")

title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an expert journalist.

    You need to come up with an interesting title for the following topic: {topic}

    Answer exactly with one title
    """,
)

essay_prompt = PromptTemplate(
    input_variables=["title", "emotion"],
    template="""
    You are an expert nonfiction writer.

    You need to write a short essay of 350 words for the following title:

    {title}

    Make sure that the essay is engaging and makes the reader feel {emotion}.

    Format the output as a JSON object with three keys: 'title', 'emotion', 'essay' and fill them with respective values
    """,
)

first_chain = title_prompt | llm | StrOutputParser()
second_chain = essay_prompt | llm | JsonOutputParser()

overall_chain = (
    first_chain 
    | (lambda title: {"title": title, "emotion": emotion}) 
    | second_chain
)

st.title("Essay Writer")

topic = st.text_input("Input Topic")
emotion = st.text_input("Input Emotion")

if topic and emotion:
    response = overall_chain.invoke({"topic": topic})
    st.write(response)
