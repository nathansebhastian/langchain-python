from langchain import hub
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(model="mistral")

prompt = hub.pull("hwchase17/react")

tools = load_tools(["wikipedia", "ddg-search", "llm-math"], llm)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

st.title("AI Agent")

question = st.chat_input("Give me a task: ")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        for response in agent_executor.stream({"input": question}):
            # Agent Action
            if "actions" in response:
                for action in response["actions"]:
                    st.write(
                        f"Calling Tool: `{action.tool}` with input `{action.tool_input}`"
                    )
            # Observation
            elif "steps" in response:
                for step in response["steps"]:
                    st.write(f"Tool Result: `{step.observation}`")
            # Final result
            elif "output" in response:
                st.write(f'Final Output: {response["output"]}')
            else:
                raise ValueError()
            st.write("---")
