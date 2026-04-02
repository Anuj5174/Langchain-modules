from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Initialize the Endpoint (Keep task as text-generation here, the wrapper handles the conversion)
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.7,
    max_new_tokens=512,
    repetition_penalty=1.1,
)

# 2. Wrap it in ChatHuggingFace to enable the 'conversational' task and support message objects
model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

print("Type 'exit' to end.")

while True:
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break
        
        # Add human message to history
        chat_history.append(HumanMessage(content=user_input))
        
        # Invoke the chat model with the history list
        result = model.invoke(chat_history)
        
        # Add AI response to history
        chat_history.append(AIMessage(content=result.content))
        print("AI: ", result.content)

print(chat_history)     
# print("\n--- Final Chat History Log ---")
# for msg in chat_history:
#     role = type(msg).__name__
#     print(f"{role}: {msg.content}")