import chromadb
import ollama

# Connect to our existing ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="my_documents"
)

# Our question
question = "What does RAG mean?"

# Create an embedding for the question
response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=question
)

question_embedding = response["embedding"]

# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

print("\nQuestion:")
print(question)

print("\nRetrieved documents:")

for document in results["documents"][0]:
    print("-", document)