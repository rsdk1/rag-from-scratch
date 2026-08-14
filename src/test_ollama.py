import chromadb
import ollama

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Delete the old collection
try:
    client.delete_collection(name="my_documents")
    print("Old collection deleted.")
except Exception:
    pass

# Create a new collection
collection = client.create_collection(
    name="my_documents"
)

documents = [
    "Playwright is a browser automation framework for testing web applications.",
    "RAG stands for Retrieval-Augmented Generation.",
    "ChromaDB is a vector database used to store and search embeddings."
]

ids = ["1", "2", "3"]

# Create embeddings using nomic-embed-text
embeddings = []

for document in documents:
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=document
    )

    embeddings.append(response["embedding"])

# Store documents + our embeddings
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents added successfully!")
print("Number of documents:", collection.count())
print("Embedding dimensions:", len(embeddings[0]))