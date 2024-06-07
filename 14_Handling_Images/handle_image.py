from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import base64

OPENAI_KEY = config("OPENAI_KEY")

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_KEY)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image = encode_image("./image-1.jpg")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can describe images in detail."),
        (
            "human",
            [
                {"type": "text", "text": "{input}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}",
                        "detail": "low",
                    },
                },
            ],
        ),
    ]
)

chain = prompt | llm
response = chain.invoke({"input": "What do you see on this image?"})

print(response.content)
