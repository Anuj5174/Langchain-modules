from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template=PromptTemplate(
    template='give me the name,age of a fictional person {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}#return a json object
)   

prompt=template.format()
result=model.invoke(prompt)
final=parser.parse(result.content)
print(final)
