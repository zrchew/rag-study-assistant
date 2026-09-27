# RAG Study Assistant

A retrieval-augmented generation (RAG) pipeline that answers questions from PDF documents, using HuggingFace embeddings, Chroma for vector storage, and Claude for answer generation.

## Setup
1. `python -m venv venv`
2. `.\venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. Set `ANTHROPIC_API_KEY` in a `.env` file
5. `python rag.py path/to/file.pdf "your question"`