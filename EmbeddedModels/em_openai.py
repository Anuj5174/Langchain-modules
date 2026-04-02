from langchain import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv("D:/LANGCHAIN/.env")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)
result = embeddings.embed_query("Hello, how are you?")
print(str(result))   