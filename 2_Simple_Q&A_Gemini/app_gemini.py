from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_GEMINI_KEY = config("GOOGLE_GEMINI_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_GEMINI_KEY)

print("Q & A With AI")
print("=============")

question = "What's the currency of Thailand?"
print("Question: " + question)

response = llm.invoke(question)

print("Answer: " + response.content)
