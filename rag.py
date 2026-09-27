"""
Phase 1 + 2: Minimal RAG pipeline over a single PDF.

Pipeline:
    PDF -> chunks -> HuggingFace embeddings -> Chroma vector store -> retrieve -> Claude answer

Usage:
    python rag_phase1_2.py path/to/your.pdf "What is the main argument in chapter 2?"

Setup (run once):
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key_here
"""

import sys
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


def build_vectorstore(pdf_path: str, persist_dir: str = "./chroma_db"):
    """Phase 1: load, chunk, embed, store."""

    # 1. Load the PDF. Each element in `pages` is one page of the document.
    print(f"Loading {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages.")

    # 2. Chunk the text. Overlap helps avoid cutting a relevant sentence
    #    exactly at a chunk boundary.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Embed each chunk with a small, fast, local HuggingFace model.
    #    This model runs on your machine — no API call, no cost.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Store the chunks + their embeddings in a local Chroma index.
    #    persist_directory means it's saved to disk and reusable next run.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Vector store built and saved to {persist_dir}.")
    return vectorstore


def inspect_retrieval(vectorstore, question: str, k: int = 4):
    """Sanity check: see what gets retrieved BEFORE trusting the LLM's answer.

    This step matters more than people expect — if retrieval pulls the
    wrong chunks, no amount of prompt engineering fixes the final answer.
    """
    results = vectorstore.similarity_search(question, k=k)
    print(f"\n--- Top {k} retrieved chunks for: '{question}' ---")
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"\n[{i}] (page {page}) {preview}...")
    return results


def answer_question(vectorstore, question: str, k: int = 4):
    """Phase 2: retrieve relevant chunks, force Claude to answer only from them."""

    retrieved_docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    # This prompt is deliberately strict: the model should say it doesn't
    # know rather than fall back on its own general knowledge. That
    # discipline is the whole point of RAG — answers should be traceable
    # to the retrieved text.
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a study assistant. Answer the question using ONLY the "
         "context provided below. If the context does not contain enough "
         "information to answer, say so explicitly rather than guessing.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ])

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return response.content, retrieved_docs


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python rag_phase1_2.py path/to/your.pdf "your question here"')
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY is not set. Set it before running "
              "the generation step, e.g.:\n  export ANTHROPIC_API_KEY=your_key_here")

    vs = build_vectorstore(pdf_path)

    # Phase 1 sanity check — always look at this before trusting Phase 2.
    inspect_retrieval(vs, question)

    # Phase 2 — generate the actual answer.
    answer, sources = answer_question(vs, question)
    print("\n=== ANSWER ===")
    print(answer)
    print("\n=== SOURCES USED ===")
    for doc in sources:
        print(f"- page {doc.metadata.get('page', '?')}")