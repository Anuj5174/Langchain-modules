from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)


##1st prompt
template1=PromptTemplate(
    template="Wrie a detailed report on {topic}",
    input_variables=['topic']
)
##2nd prompt
template2=PromptTemplate(
    template="Wrie a 5line summary on {text}",
    input_variables=['text']
)

prompt1=template1.invoke({'topic':'blackhole'})
result=model.invoke(prompt1)

prompt2=template2.invoke({'text':result.content})
result2=model.invoke(prompt2)

print(result2.content)