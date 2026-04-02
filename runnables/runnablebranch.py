from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableBranch
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

def word(text):
    return len(text.split())


passthrough=RunnablePassthrough()

parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Write a detailed variable on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Summarize the following {text}',
    input_variables=['text']
)


report_gen_chain=RunnableSequence(
    prompt1,
    model,
    parser
)

branch_chain=RunnableBranch(
    (lambda x:len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final=RunnableSequence(
    report_gen_chain,
    branch_chain
)

print(final.invoke({'topic':'cats'}))


