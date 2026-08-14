from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import chromadb
import ollama

app = FastAPI(title="Local Stock RAG")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="stock_documents"
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Local Stock RAG</title>
    </head>

    <body>

        <h1>📊 Local Stock RAG</h1>

        <p>HDFC Bank Research Assistant</p>

        <form method="post" action="/ask">

            <textarea
                name="question"
                rows="5"
                cols="70"
                placeholder="Ask something about HDFC Bank..."
                required
            ></textarea>

            <br><br>

            <button type="submit">
                Ask RAG
            </button>

        </form>

    </body>
    </html>
    """


@app.post("/ask", response_class=HTMLResponse)
def ask_rag(question: str = Form(...)):

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=question
    )

    question_embedding = response["embedding"]

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

    context_parts = []

    for i, document in enumerate(documents):

        metadata = metadatas[i]

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

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

    prompt = f"""
You are a financial research assistant.

Use ONLY the information provided in the research sources.

Do not invent facts, prices, dates, financial results,
or events.

Do not treat information in the PDF as live market data.

If the requested information is not available,
say so clearly.

If the question asks for current, today's, latest,
or live information and the PDF does not contain
live data, explain that the document does not contain
live market data.

RESEARCH SOURCES:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    answer = ollama.generate(
        model="llama3.2",
        prompt=prompt,
        options={
            "temperature": 0.2,
            "num_predict": 1000
        }
    )

    answer_text = answer["response"]

    sources_html = ""

    seen_sources = set()

    for metadata in metadatas:

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        source_key = (source, page)

        if source_key not in seen_sources:

            sources_html += (
                f"<li>{source} | PDF Page {page}</li>"
            )

            seen_sources.add(source_key)

    return f"""
    <!DOCTYPE html>
    <html>

    <head>
        <title>Local Stock RAG</title>
    </head>

    <body>

        <h1>📊 Local Stock RAG</h1>

        <h2>Question</h2>

        <p>{question}</p>

        <h2>RAG Answer</h2>

        <pre>{answer_text}</pre>

        <h2>Retrieved Sources</h2>

        <ul>
            {sources_html}
        </ul>

        <a href="/">
            ← Ask another question
        </a>

    </body>

    </html>
    """