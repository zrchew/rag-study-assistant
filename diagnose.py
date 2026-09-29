"""
Retrieval debugger. Ranks EVERY chunk in the index for a question and tells
you where a phrase you KNOW is in the answer lands. Costs nothing (no Claude
call) and answers "did retrieval find it, or did the model miss it?".

Run from the project root, after build_index.py:

    python diagnose.py "What does SCALE stand for?" "Context-Aware Learning Engine"
    python diagnose.py "Did the paper test on real Vietnamese textbooks?" "important future work"
    python diagnose.py "your question" "your phrase" --top 20

Answers only use the top k chunks (k=6 by default), so a phrase whose best
chunk ranks #12 is retrieved by --k 12 but not by the defaults.

Matching ignores case, spaces, line breaks, hyphens and punctuation, so
differences in how the PDF text was extracted can't hide a match.
"""

import argparse
import re

from chat import load_text_index


def normalize(text):
    """Lowercase and keep only letters and digits."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def one_line(doc, width=80):
    return " ".join(doc.page_content.split())[:width]


def page_of(doc):
    return doc.metadata.get("page", -1) + 1  # 1-based, matches page_N.png


def main():
    parser = argparse.ArgumentParser(description="Show where a phrase ranks in retrieval.")
    parser.add_argument("question", help="the question to search with")
    parser.add_argument("phrase", help="text that should appear in a chunk that answers it")
    parser.add_argument("--top", type=int, default=10,
                        help="how many top-ranked chunks to list (default 10)")
    args = parser.parse_args()

    vectorstore = load_text_index()
    try:
        total = vectorstore._collection.count()
    except Exception:
        total = 500
    docs = vectorstore.similarity_search(args.question, k=total)  # rank every chunk

    target = normalize(args.phrase)
    hits = [(rank, d) for rank, d in enumerate(docs, start=1)
            if target in normalize(d.page_content)]
    hit_ranks = {rank for rank, _ in hits}

    print(f"\nQuestion: {args.question!r}")
    print(f"Looking for: {args.phrase!r}\n")
    print(f"Top {args.top} chunks:")
    for rank, doc in enumerate(docs[:args.top], start=1):
        flag = "<<< HAS PHRASE" if rank in hit_ranks else ""
        print(f"{rank:>4}  p.{page_of(doc):<3} {flag:<15} {one_line(doc)}")

    print(f"\nChecked all {len(docs)} chunks in the index.")
    if not hits:
        print("No single chunk contains this phrase, so this is NOT a ranking problem.")
        print("Either it is cut across a chunk boundary, or the PDF text extraction "
              "garbled it. Try a shorter phrase (2-3 distinctive words) to find out.")
    else:
        best_rank, best_doc = hits[0]
        pages = sorted({page_of(d) for _, d in hits})
        print(f"{len(hits)} chunk(s) contain it, on page(s) {pages}.")
        print(f"Best-ranked: #{best_rank} of {len(docs)}  (p.{page_of(best_doc)})  {one_line(best_doc)}")
        print(f"Answers use only the top k chunks, so k must be at least {best_rank} "
              "to retrieve it (default k=6).")


if __name__ == "__main__":
    main()