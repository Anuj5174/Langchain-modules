from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.7,
    max_new_tokens=512,
    repetition_penalty=1.1,
)
model = ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='Generate a detailed report on topic  {topic}',
    input_variables=['topic'],
)

prompt2=PromptTemplate(
    template='Genarate a  5 point summary of {report}',
    input_variables=['report'],
)

parser=StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
result=chain.invoke({'topic':'cricket'})
print(result)

chain.get_graph().print_ascii()