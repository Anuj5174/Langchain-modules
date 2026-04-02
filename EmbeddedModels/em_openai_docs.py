from langchain import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv("D:/LANGCHAIN/.env")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)
documents=[
    "The cat sat on the mat.",
    "The dog chased the ball.",
    "The bird flew in the sky."
]
result = embeddings.embed_documents(documents)
print(str(result))   