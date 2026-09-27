"""
Multimodal RAG extension.

Adds image-based retrieval for tables/charts/diagrams that exist as pictures
in the PDF (not extractable text), using CLIP -- a model that embeds both
images and text into the SAME vector space.

Key design point (per your requirement): images are embedded ONCE, at
indexing time. At query time, only your QUESTION gets embedded (cheap,
text-only) and compared against the already-computed image embeddings.
No image encoding happens per query -- only the single best-matching
image (if it clears a similarity threshold) gets sent to Claude.

Usage:
    python rag_multimodal.py path/to/your.pdf "your question here"

New dependency vs rag_phase1_2.py:
    pip install pymupdf
"""

import sys
import os
import base64

import fitz  # PyMuPDF -- renders PDF pages to images, no external binary needed
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

IMAGE_DIR = "./page_images"
# Cosine similarity cutoff for "is this image actually relevant to the
# question, or should we skip attaching an image at all". Tune this by
# testing -- if irrelevant images keep getting attached, raise it; if a
# clearly relevant table/chart isn't being picked up, lower it.
IMAGE_MATCH_THRESHOLD = 0.22


def render_pages_as_images(pdf_path: str, out_dir: str = IMAGE_DIR):
    """One-time step: render every PDF page to a PNG file on disk."""
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        out_path = os.path.join(out_dir, f"page_{i + 1}.png")
        pix.save(out_path)
        paths.append(out_path)
    doc.close()
    print(f"Rendered {len(paths)} page images to {out_dir}")
    return paths


def build_clip_image_index(image_paths):
    """Embed every page image ONCE with CLIP. This is the only place
    images get encoded in this whole script."""
    clip_model = SentenceTransformer("clip-ViT-B-32")
    images = [Image.open(p) for p in image_paths]
    print("Embedding page images with CLIP (one-time cost)...")
    image_embeddings = clip_model.encode(
        images, convert_to_tensor=True, show_progress_bar=True
    )
    return clip_model, image_embeddings


def find_best_matching_image(clip_model, image_embeddings, image_paths, question):
    """Embed the QUESTION only -- fast, text-only -- and compare it against
    the image embeddings computed once in build_clip_image_index. No image
    encoding happens in this function."""
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


def build_text_vectorstore(pdf_path: str, persist_dir: str = "./chroma_db"):
    """Same text pipeline as rag_phase1_2.py, with the tuned chunk settings
    from earlier (larger chunks, more overlap, since the source is a
    slide-style PDF with sparse text per page)."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return vectorstore


def answer_question_multimodal(vectorstore, clip_model, image_embeddings, image_paths, question, k=6):
    # 1. Text retrieval -- unchanged from your original script.
    text_docs = vectorstore.similarity_search(question, k=k)
    text_context = "\n\n---\n\n".join(d.page_content for d in text_docs)

    # 2. Image retrieval -- only the question gets encoded here.
    best_image_path, score = find_best_matching_image(
        clip_model, image_embeddings, image_paths, question
    )
    print(f"Best image match: {best_image_path} (similarity={score:.3f})")

    # 3. Build the Claude message: text context always included,
    #    image only attached if it cleared the relevance threshold.
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
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_to_base64(best_image_path),
            },
        })

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    response = llm.invoke([HumanMessage(content=content)])
    return response.content, text_docs, best_image_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python rag_multimodal.py path/to/your.pdf "your question here"')
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    print("Building text vector store...")
    vs = build_text_vectorstore(pdf_path)

    print("Rendering pages to images...")
    image_paths = render_pages_as_images(pdf_path)

    clip_model, image_embeddings = build_clip_image_index(image_paths)

    answer, text_docs, best_image = answer_question_multimodal(
        vs, clip_model, image_embeddings, image_paths, question
    )

    print("\n=== ANSWER ===")
    print(answer)
    print("\n=== TEXT SOURCES ===")
    for d in text_docs:
        print(f"- page {d.metadata.get('page', '?')}")
    if best_image:
        print(f"\n=== IMAGE USED ===\n{best_image}")
    else:
        print("\n(No page image cleared the relevance threshold for this question.)")