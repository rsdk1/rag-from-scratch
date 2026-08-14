import chromadb
import ollama

# ==========================================
# 1. CONNECT TO CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="stock_documents"
)


# ==========================================
# 2. QUESTION
# ==========================================

question = "What are the major risks of HDFC Bank?"


# ==========================================
# 3. CREATE QUESTION EMBEDDING
# ==========================================

response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=question
)

question_embedding = response["embedding"]


# ==========================================
# 4. RETRIEVE RELEVANT DOCUMENTS
# ==========================================

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=5,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]


# ==========================================
# 5. BUILD CONTEXT FOR LLAMA
# ==========================================

context_parts = []

for i, document in enumerate(documents):

    metadata = metadatas[i]

    source = metadata.get(
        "source",
        "Unknown"
    )

    page = metadata.get(
        "page",
        "Unknown"
    )

    context_parts.append(
        f"""
SOURCE {i + 1}

Document: {source}
PDF Page: {page}

Content:
{document}
"""
    )

context = "\n".join(context_parts)


# ==========================================
# 6. PROMPT LLAMA
# ==========================================

prompt = f"""
You are a financial research assistant.

You are answering questions using a research PDF
provided by the user.

IMPORTANT RULES:

1. Use ONLY the information contained in the
   provided sources.

2. Do NOT invent facts, numbers, prices, dates,
   financial results, or events.

3. Do NOT treat information from the PDF as live
   market information.

4. If the user asks for current, today's, latest,
   or live information and the provided document
   does not contain live data, clearly say:

"The provided document does not contain live market
data. It only contains information available in
the research report."

5. If the PDF contains a historical or
   report-date value, clearly explain that it is
   from the report and not necessarily current.

6. If the requested information is not available
   in the provided sources, say:

"This information is not available in the
provided document."

7. Do not confuse the report date with a PDF page
   number.

8. Give a clear and structured answer.

9. You may summarize information from multiple
   retrieved sources.

10. Do not make investment recommendations unless
    the provided document explicitly contains one.

RESEARCH SOURCES:

{context}

USER QUESTION:

{question}

ANSWER:
"""


# ==========================================
# 7. GENERATE ANSWER
# ==========================================

answer = ollama.generate(
    model="llama3.2",
    prompt=prompt,
    options={
        "temperature": 0.2,
        "num_predict": 1000
    }
)


# ==========================================
# 8. DISPLAY ANSWER
# ==========================================

print("\n================================")
print("QUESTION")
print("================================")

print(question)


print("\n================================")
print("RAG ANSWER")
print("================================")

print(answer["response"])


# ==========================================
# 9. DISPLAY SOURCES
# ==========================================

print("\n================================")
print("RETRIEVED SOURCES")
print("================================")

seen_sources = set()

for metadata in metadatas:

    source = metadata.get(
        "source",
        "Unknown"
    )

    page = metadata.get(
        "page",
        "Unknown"
    )

    source_key = (source, page)

    if source_key not in seen_sources:

        print(
            f"- {source} | PDF Page {page}"
        )

        seen_sources.add(source_key)