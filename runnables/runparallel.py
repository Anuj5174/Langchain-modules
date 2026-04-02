from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
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

prompt1=PromptTemplate(
    template='Generate a tweet on  {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='Generate a linkedin post on  {topic}',
    input_variables=['topic']
)


parser=StrOutputParser()

parallel=RunnableParallel(
    {
        "tweet":RunnableSequence(prompt1,model,parser),
        "linkedin":RunnableSequence(prompt2,model,parser)
    }
)

print(parallel.invoke({'topic':'AI'}))