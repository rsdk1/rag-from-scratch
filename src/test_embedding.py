import ollama

text = "RAG retrieves relevant information from documents before generating an answer."

response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=text
)

embedding = response["embedding"]

print("Embedding created successfully!")
print("Number of dimensions:", len(embedding))
print("First 10 values:", embedding[:10])