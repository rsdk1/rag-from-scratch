import chromadb
import ollama

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="stock_documents"
)

# Ask a question
question = "What are the major risks of HDFC Bank?"

# Create embedding for the question
response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=question
)

question_embedding = response["embedding"]

# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

print("\nQuestion:")
print(question)

print("\nRetrieved documents:")

for i, document in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]

    print(f"\n--- Result {i + 1} ---")
    print("Page:", metadata["page"])
    print(document)