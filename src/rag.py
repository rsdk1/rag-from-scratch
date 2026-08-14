import chromadb
import ollama

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="my_documents"
)

# User question
question = "What does RAG mean?"

# Create embedding for the question
response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=question
)

question_embedding = response["embedding"]

# Retrieve relevant documents
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

# Combine retrieved documents into context
context = "\n\n".join(results["documents"][0])

# Create prompt for Llama
prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided context."
"""

# Ask Llama
answer = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer["message"]["content"])