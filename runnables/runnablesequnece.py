from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    max_new_tokens=50,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template='Write a joke on {topic}',
    input_variables=['topic']
)

output_parser=StrOutputParser()

chain=RunnableSequence(
    prompt,model,output_parser
)

print(chain.invoke({'topic':'cats'}))