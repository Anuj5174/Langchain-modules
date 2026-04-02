from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
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

passthrough=RunnablePassthrough()

prompt1=PromptTemplate(
    template='Write a joke on {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='Explain joke {joke}',
    input_variables=['joke']
)

parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt1,model,parser)

parallel=RunnableParallel(
    {
        'joke' : RunnablePassthrough(),
        'explanation':RunnableSequence(prompt2,model,parser)
    }
)

final = RunnableSequence(joke_gen_chain,parallel)


print(final.invoke({'topic':'cats'}))