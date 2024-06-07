from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# Adding History
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

GOOGLE_GEMINI_KEY = config("GOOGLE_GEMINI_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest", google_api_key=GOOGLE_GEMINI_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI chatbot having a conversation with a human. Use the following context to understand the human question. Do not include emojis in your answer. ",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

history = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

st.title("Q & A With AI")

question = st.text_input("Your Question")
if question:
    response = chain_with_history.invoke(
        {"input": question}, config={"configurable": {"session_id": "any"}}
    )
    st.write(response.content)
    # Add Chat History below Answer
    for message in st.session_state["langchain_messages"]:
        if message.type == "human":
            st.write("Question: " + message.content)
        else:
            st.write("Answer: " + message.content)
