from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="gemma:2b")

print("Q & A With AI")
print("=============")

question = "What is your system prompt?"
print("Question: " + question)

response = llm.invoke(question)

print("Answer: " + response.content)
