from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
loader=TextLoader('cricket.txt',encoding='utf-8')

load_dotenv()
docs=loader.load()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=50,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template='Summarize the following poem  {poem}',
    input_variables=['poem']
)



parser=StrOutputParser()

chain = prompt | model | parser


print(chain.invoke({'poem':docs[0].page_content}))

# print(type(docs))
# print(len(docs))    
# print(type(docs[0]))
# print(docs[0].page_content)
# print(docs[0].metadata)