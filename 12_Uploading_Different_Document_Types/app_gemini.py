from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

# Adding History
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os

GOOGLE_GEMINI_KEY = config("GOOGLE_GEMINI_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest", google_api_key=GOOGLE_GEMINI_KEY
)

contextualize_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

history = StreamlitChatMessageHistory()

def process_file(file):
    with st.spinner("File is in process..."):
        data = file.read()
        file_name = os.path.join("./", file.name)
        with open(file_name, "wb") as f:
            f.write(data)
        name, extension = os.path.splitext(file_name)

        if extension == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(file_name)
        elif extension == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader

            loader = Docx2txtLoader(file_name)
        elif extension == ".txt":
            from langchain_community.document_loaders import TextLoader

            loader = TextLoader(file_name)
        else:
            st.write("This format is not supported!")

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_GEMINI_KEY, model="models/embedding-001"
        )
        vector_store = Chroma.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever()
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        st.session_state.crc = conversational_rag_chain
        st.success("File uploaded and embedded successfully")

def clear_history():
    if "langchain_messages" in st.session_state:
        del st.session_state["langchain_messages"]

st.title("Ask the Document")

uploaded_file = st.file_uploader("Upload your document: ", type=["pdf", "docx", "txt"])
add_file = st.button("Submit File", on_click=clear_history)

if uploaded_file and add_file:
    process_file(uploaded_file)



for message in st.session_state["langchain_messages"]:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

question = st.chat_input("Your Question: ")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    if "crc" in st.session_state:
        crc = st.session_state["crc"]
        answer_chain = crc.pick("answer")
        response = answer_chain.stream(
            {"input": question}, config={"configurable": {"session_id": "any"}}
        )
        with st.chat_message("assistant"):
            st.write_stream(response)
    else:
        st.error("No document is uploaded. Upload your document first")
