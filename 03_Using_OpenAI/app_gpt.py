from decouple import config
from langchain_openai import ChatOpenAI

OPENAI_KEY = config("OPENAI_KEY")

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_KEY)

print("Q & A With AI")
print("=============")

question = "What's the currency of Thailand?"
print("Question: " + question)

response = llm.invoke(question)

print("Answer: " + response.content)
