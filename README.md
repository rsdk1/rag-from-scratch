# Local RAG Financial Research Assistant

A local Retrieval-Augmented Generation (RAG) system for querying financial research documents using natural language.

The project uses local LLMs and embeddings through Ollama, with ChromaDB for semantic document retrieval and FastAPI for a local web interface.

## 🚀 Project Overview

This project demonstrates an end-to-end local RAG pipeline:

PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Semantic Retrieval → Llama 3.2 → Grounded Answer

Currently, the system is tested using an HDFC Bank research report.

## 🧠 Technologies Used

- Python
- Retrieval-Augmented Generation (RAG)
- Ollama
- Llama 3.2
- nomic-embed-text
- ChromaDB
- FastAPI
- PDF text extraction
- Semantic search
- Vector embeddings
- Prompt engineering

## 🏗️ Architecture

```text
                 Research PDF
                      │
                      ▼
              PDF Text Extraction
                      │
                      ▼
                 Text Chunking
                      │
                      ▼
              nomic-embed-text
                      │
                      ▼
                  ChromaDB
               Vector Database
                      │
                      │
User Question ───────┘
      │
      ▼
Question Embedding
      │
      ▼
Semantic Retrieval
      │
      ▼
Relevant PDF Context
      │
      ▼
Llama 3.2 via Ollama
      │
      ▼
Grounded RAG Answer
      │
      ▼
Source + PDF Page