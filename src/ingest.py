import pymupdf
import chromadb
import ollama

PDF_PATH = "documents/HDFC_Bank_Research_Report.pdf"

# -----------------------------
# 1. Read PDF
# -----------------------------

doc = pymupdf.open(PDF_PATH)

pages = []

for page_number, page in enumerate(doc):
    text = page.get_text().strip()

    if text:
        pages.append({
            "page": page_number + 1,
            "text": text
        })

doc.close()

print("Pages extracted:", len(pages))


# -----------------------------
# 2. Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Delete previous test collection
try:
    client.delete_collection("stock_documents")
    print("Old collection deleted.")
except Exception:
    pass

collection = client.create_collection(
    name="stock_documents"
)


# -----------------------------
# 3. Create chunks
# -----------------------------

chunk_size = 1500
overlap = 200

documents = []
embeddings = []
ids = []
metadatas = []

chunk_number = 0

for page in pages:

    text = page["text"]
    page_number = page["page"]

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunk_id = f"hdfcbank_{chunk_number}"

            print(f"Creating embedding: {chunk_id}")

            # Create embedding using Ollama
            response = ollama.embeddings(
                model="nomic-embed-text",
                prompt=chunk
            )

            embedding = response["embedding"]

            documents.append(chunk)
            embeddings.append(embedding)
            ids.append(chunk_id)

            metadatas.append({
                "company": "HDFC Bank",
                "ticker": "HDFCBANK",
                "source": PDF_PATH,
                "page": page_number,
                "chunk": chunk_number
            })

            chunk_number += 1

        start = end - overlap


# -----------------------------
# 4. Store in ChromaDB
# -----------------------------

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("\n-----------------------------")
print("Ingestion completed!")
print("-----------------------------")

print("Documents stored:", collection.count())
print("Embedding dimensions:", len(embeddings[0]))