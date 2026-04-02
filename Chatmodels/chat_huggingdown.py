from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv("D:/LANGCHAIN/.env")

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100,"temperature":0.5,"max_new_tokens":100},
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Who is starring in the movie Animal?")
print(result.content)