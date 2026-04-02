from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "Lionel Messi is an Argentine footballer widely regarded as one of the greatest players of all time.",
    "Serena Williams is an American tennis player who has won 23 Grand Slam singles titles.",
    "Usain Bolt is a Jamaican sprinter known for his world records in the 100m and 200m events.",
    "Roger Federer is a Swiss tennis player known for his elegant playing style and 20 Grand Slam titles."
]

query = "Tell me about a famous footballer"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)