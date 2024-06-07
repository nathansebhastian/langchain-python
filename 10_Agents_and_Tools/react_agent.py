from decouple import config
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI

OPENAI_KEY = config("OPENAI_KEY")

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_KEY)

prompt = hub.pull("hwchase17/react")
prompt.pretty_print()

tools = load_tools(["wikipedia", "ddg-search", "llm-math"], llm)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools)

question = input("Give me a task: ")

for response in agent_executor.stream({"input": question}):
    # Agent Action
    if "actions" in response:
        for action in response["actions"]:
            print(f"Calling Tool: `{action.tool}` with input `{action.tool_input}`")
    # Observation
    elif "steps" in response:
        for step in response["steps"]:
            print(f"Tool Result: `{step.observation}`")
    # Final result
    elif "output" in response:
        print(f'Final Output: {response["output"]}')
    else:
        raise ValueError()
    print("---")
