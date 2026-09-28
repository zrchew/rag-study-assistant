"""
Indexing script -- run this ONCE per PDF.

Builds and saves to disk:
  1. The text vector store (Chroma, persisted to ./chroma_db)
  2. The CLIP image embeddings for every page (saved to ./index_store)

After running this, use chat.py to ask as many questions as you want
without re-doing any of this setup work.

Usage:
    python build_index.py path/to/your.pdf
"""

import sys
import os
import json
import shutil

import fitz
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()  # picks up HF_TOKEN (if set) for faster/authenticated model downloads

CHROMA_DIR = "./chroma_db"
IMAGE_DIR = "./page_images"
INDEX_STORE_DIR = "./index_store"


def render_pages_as_images(pdf_path: str, out_dir: str = IMAGE_DIR):
    # Start clean so pages from a previously indexed PDF don't linger.
    shutil.rmtree(out_dir, ignore_errors=True)
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


def build_and_save_text_index(pdf_path: str):
    # Chroma.from_documents() APPENDS to an existing store. Without this
    # cleanup, re-running on the same PDF duplicates every chunk, and
    # indexing a different PDF mixes the two documents together.
    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
        except PermissionError:
            print(f"Could not delete the old {CHROMA_DIR}. Close chat.py (or anything "
                  "else using it) and try again.")
            sys.exit(1)
        print(f"Removed old text index at {CHROMA_DIR}")

    print("Building text vector store...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # persist_directory means this is saved to disk -- chat.py will load it
    # back later without re-embedding anything.
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print(f"Text index saved to {CHROMA_DIR}")


def build_and_save_image_index(pdf_path: str):
    image_paths = render_pages_as_images(pdf_path)

    print("Embedding page images with CLIP (one-time cost)...")
    clip_model = SentenceTransformer("clip-ViT-B-32")
    images = [Image.open(p) for p in image_paths]
    image_embeddings = clip_model.encode(images, convert_to_tensor=True, show_progress_bar=True)

    os.makedirs(INDEX_STORE_DIR, exist_ok=True)
    torch.save(image_embeddings, os.path.join(INDEX_STORE_DIR, "image_embeddings.pt"))
    with open(os.path.join(INDEX_STORE_DIR, "image_paths.json"), "w") as f:
        json.dump(image_paths, f)
    print(f"Image index saved to {INDEX_STORE_DIR}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_index.py path/to/your.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    build_and_save_text_index(pdf_path)
    build_and_save_image_index(pdf_path)
    print("\nIndexing complete. Run: python chat.py")