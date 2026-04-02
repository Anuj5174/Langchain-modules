from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
loader=WebBaseLoader('https://www.amazon.in/Lenovo-Smartchoice-WUXGA-OLED-Microsoft-83CV00B3IN/dp/B0FCXVS3NK?pf_rd_p=ee7f405c-cd76-4264-bebd-2502f5a7c5a6&pf_rd_r=Z138RT6DFJRDBP1FZ0DV&ref_=AI_Premium_B0FCXVS3NK&th=1')

docs=loader.load()
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
    template='Answer the following question {question} from the following text  {text}',
    input_variables=['question','text']
)

chain=prompt|model|StrOutputParser()

print(chain.invoke({'question':'What is the price of the product?','text':docs[0].page_content[:6000]}))

# print(len(docs))

# print(docs[0].page_content)