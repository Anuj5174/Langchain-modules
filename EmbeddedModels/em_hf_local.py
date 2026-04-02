from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv("D:/LANGCHAIN/.env")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

text = "Delhi is the capital of india"

result = embeddings.embed_query(text)
print(str(result))