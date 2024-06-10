from decouple import config
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug

set_debug(True)

OPENAI_KEY = config("OPENAI_KEY")

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_KEY)

title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an expert journalist.

    You need to come up with an interesting title for the following topic: {topic}

    Answer exactly with one title
    """,
)

essay_prompt = PromptTemplate(
    input_variables=["title"],
    template="""
    You are an expert nonfiction writer.

    You need to write a short essay of 350 words for the following title:

    {title}

    Make sure that the essay is engaging and makes the reader feel excited.
    """,
)

first_chain = title_prompt | llm | StrOutputParser()
second_chain = essay_prompt | llm

overall_chain = first_chain | second_chain

st.title("Essay Writer")

topic = st.text_input("Input Topic")

if topic:
    response = overall_chain.invoke({"topic": topic})
    st.write(response.content)
