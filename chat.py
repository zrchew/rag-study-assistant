"""
Chat script -- run this as many times / for as many questions as you want.

Loads the text and image indexes built by build_index.py (no re-embedding
of the document happens here) and lets you ask questions in a loop,
including follow-ups that refer back to earlier turns in the conversation.

Usage:
    python build_index.py path/to/your.pdf   (run once)
    python chat.py                            (run repeatedly)
"""

import os
import json
import base64

from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

CHROMA_DIR = "./chroma_db"
INDEX_STORE_DIR = "./index_store"
IMAGE_MATCH_THRESHOLD = 0.22
# How many past question/answer pairs to include as conversation memory.
# Keeping this small keeps token usage (and cost) predictable.
MAX_HISTORY_TURNS = 4


def load_text_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Passing persist_directory here LOADS the existing store from disk --
    # this does not re-embed anything, unlike Chroma.from_documents().
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def load_image_index():
    clip_model = SentenceTransformer("clip-ViT-B-32")
    image_embeddings = torch.load(os.path.join(INDEX_STORE_DIR, "image_embeddings.pt"))
    with open(os.path.join(INDEX_STORE_DIR, "image_paths.json")) as f:
        image_paths = json.load(f)
    return clip_model, image_embeddings, image_paths


def find_best_matching_image(clip_model, image_embeddings, image_paths, question):
    question_embedding = clip_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, image_embeddings)[0]
    best_idx = int(torch.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score >= IMAGE_MATCH_THRESHOLD:
        return image_paths[best_idx], best_score
    return None, best_score


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def answer_question(vectorstore, clip_model, image_embeddings, image_paths, question, history, k=6):
    text_docs = vectorstore.similarity_search(question, k=k)
    text_context = "\n\n---\n\n".join(d.page_content for d in text_docs)

    best_image_path, score = find_best_matching_image(clip_model, image_embeddings, image_paths, question)

    content = [
        {
            "type": "text",
            "text": (
                "Answer the question using the context below and, if an "
                "image is provided, the image as well. If neither contains "
                "enough information, say so rather than guessing.\n\n"
                f"Context:\n{text_context}\n\nQuestion: {question}"
            ),
        }
    ]
    if best_image_path:
        content.insert(0, {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image_to_base64(best_image_path)},
        })

    # `history` carries prior turns so follow-up questions ("what about
    # page 21?") have something to refer back to.
    messages = history + [HumanMessage(content=content)]

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    response = llm.invoke(messages)
    return response.content, text_docs, best_image_path


if __name__ == "__main__":
    if not os.path.exists(CHROMA_DIR) or not os.path.exists(INDEX_STORE_DIR):
        print("No index found. Run this first: python build_index.py path/to/your.pdf")
        exit(1)

    print("Loading text index...")
    vectorstore = load_text_index()
    print("Loading image index...")
    clip_model, image_embeddings, image_paths = load_image_index()

    print("\nReady. Ask a question, or type 'exit' to quit.\n")

    history = []  # list of HumanMessage / AIMessage, oldest first

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer, text_docs, best_image = answer_question(
            vectorstore, clip_model, image_embeddings, image_paths, question, history
        )

        print(f"\nClaude: {answer}\n")
        pages = sorted(set(d.metadata.get("page", "?") for d in text_docs))
        print(f"(sources: pages {pages}" + (f", image: {best_image})" if best_image else ")"))
        print()

        # Keep the conversation history from growing unbounded -- only the
        # answer text goes back in (not the image/context blob) to keep
        # token usage down on later turns.
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        history = history[-(MAX_HISTORY_TURNS * 2):]