from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
import os
from pydantic import BaseModel , Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
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
parser=StrOutputParser()

class feedback(BaseModel):
    sentiment:Literal['Positive','Negative']=Field(description='sentiment of the feedback')

pydantic_parser=PydanticOutputParser(pydantic_object=feedback)

prompt1=PromptTemplate(
    template='classify the sentiment of the following feedbacktext into pos or neg  {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':pydantic_parser.get_format_instructions()}
)


classifier_chain = prompt1 | model | pydantic_parser

prompt2=PromptTemplate(
    template='Write an appropriate response to a positive feedback {feedback}',
    input_variables=['feedback']
) 
prompt3=PromptTemplate(
    template='Write an appropriate response to a negative feedback {feedback}',
    input_variables=['feedback']  
) 
branch_chain=RunnableBranch(
 (lambda x:x.sentiment=='Positive',prompt2|model|parser),
 (lambda x:x.sentiment=='Negative',prompt3|model|parser),
 RunnableLambda(lambda x:"Couldnot find any sentiment")
)
chain = classifier_chain|branch_chain
result=chain.invoke({'feedback':'This is a beautiful  iphone'})
print(result)
