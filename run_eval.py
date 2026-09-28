"""
Evaluation runner.

Asks a fixed list of questions through the same pipeline chat.py uses and
saves the answers to CSV, so you can mark each one Pass / Partial / Fail
by hand. Also lets you change retrieval settings from the command line so
you can compare configurations for your README.

Run from the project root, after build_index.py:

    python run_eval.py                        # one run at the defaults
    python run_eval.py --k 10                 # different number of retrieved chunks
    python run_eval.py --threshold 0.30       # different image-match threshold
    python run_eval.py --sweep                # one-at-a-time sweep of k and threshold
    python run_eval.py --label chunk800       # tag a run (see note below)

Cost: every question is one Claude call. One run = 11 calls, a sweep = 55.

The --label flag is for settings that need a re-index (chunk size, overlap):
change the values in build_index.py, rebuild, then run with e.g.
--label chunk800 so the files and summary row say which index was used.

Output (in ./eval_results):
    <label_>k<k>_thr<threshold>.csv   one file per configuration, with a blank
                                      Result column for you to fill in
    eval_summary.csv                  one row per configuration run (appended)
"""

import argparse
import csv
import os
import sys
import time

from langchain_core.messages import HumanMessage, AIMessage

import chat  # reuses load_text_index, load_image_index, answer_question, ...

DEFAULT_K = 6
DEFAULT_THRESHOLD = 0.22
SWEEP_K_VALUES = [3, 6, 10]
SWEEP_THRESHOLD_VALUES = [0.15, 0.22, 0.30]
OUT_DIR = "eval_results"

# "expected" is only a label copied into the CSV so you can compare it with
# the bot's answer when marking. The script never checks answers itself.
# "followup" means the question only makes sense after the one before it, so
# that previous question and answer are passed in as conversation history.
# Every other question starts with an empty history.
QUESTIONS = [
    {"q": "What is the name of the AI tutoring system in the paper?",
     "expected": "DeepEdu-v1"},
    {"q": "What does SCALE stand for?",
     "expected": "Self-improving Context-Aware Learning Engine"},
    {"q": "What does SCR stand for?",
     "expected": "Similarity Chunk Rolling"},
    {"q": "Which Vietnamese law does the paper say cloud assistants like ChatGPT violate?",
     "expected": "Decree 53"},
    {"q": "How many fewer retrieval calls does SCR make?",
     "expected": "7.7x fewer"},
    {"q": "And roughly how much does that cut prefill latency (TTFT)?",
     "expected": "About 35%", "followup": True},
    {"q": "Which GPU was used for the experiments?",
     "expected": "A single NVIDIA H100 (80 GB)"},
    {"q": "In Table VI, what does TokenSelect score on R.KV?",
     "expected": "91.0%"},
    {"q": "What about SCR with Lmax = 4096?",
     "expected": "98.0%", "followup": True},
    {"q": "Did the paper test on real Vietnamese textbooks?",
     "expected": "No - listed as future work"},
    # Figure-only question: the number appears in a chart, not in the running
    # text, so this is the one that exercises image retrieval. Delete it if you
    # would rather keep the set to ten.
    {"q": "In Figure 1 panel (c), what is the TPOT for the full Playbook?",
     "expected": "About 0.0086 s (figure only)"},
]

DETAIL_FIELDS = ["config", "n", "question", "expected", "answer", "text_pages",
                 "image_used", "image_score", "seconds", "result", "notes"]
SUMMARY_FIELDS = ["label", "k", "threshold", "questions", "images_attached",
                  "avg_seconds", "errors", "manual_pass"]


def clean(text):
    """Flatten whitespace so each answer fits on one spreadsheet row."""
    return " ".join(str(text).split())


def content_to_text(content):
    """ChatAnthropic normally returns a string but can return a list of blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def write_csv(path, fieldnames, rows, append=False):
    """Write rows to CSV. utf-8-sig so Excel shows symbols like x correctly."""
    is_new = not os.path.exists(path)
    mode = "a" if append else "w"
    try:
        with open(path, mode, newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not append or is_new:
                writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        print(f"\nCould not write {path}. If it is open in Excel, close it and run again.")
        sys.exit(1)


def run_config(k, threshold, label, vectorstore, clip_model, image_embeddings, image_paths):
    # chat.find_best_matching_image reads this module-level value at call time.
    chat.IMAGE_MATCH_THRESHOLD = threshold
    config_name = f"{label + '_' if label else ''}k{k}_thr{threshold:.2f}"
    print(f"\n=== {config_name} ===")

    rows = []
    prev_q = prev_a = None
    for n, item in enumerate(QUESTIONS, start=1):
        history = []
        if item.get("followup") and prev_q is not None:
            history = [HumanMessage(content=prev_q), AIMessage(content=prev_a)]

        start = time.perf_counter()
        try:
            content, text_docs, image_used = chat.answer_question(
                vectorstore, clip_model, image_embeddings, image_paths,
                item["q"], history, k=k)
            answer = content_to_text(content)
            # +1: PyPDFLoader counts pages from 0; this matches page_N.png
            pages = sorted({d.metadata.get("page", -1) + 1 for d in text_docs})
        except Exception as e:  # keep going so one failure doesn't lose the run
            answer, pages, image_used = f"ERROR: {e}", [], None
        seconds = time.perf_counter() - start

        # Log how similar the question was to its best page image, so you can
        # see which scores separate figure questions from text questions.
        _, score = chat.find_best_matching_image(
            clip_model, image_embeddings, image_paths, item["q"])

        rows.append({
            "config": config_name,
            "n": n,
            "question": item["q"],
            "expected": item["expected"],
            "answer": clean(answer),
            "text_pages": " ".join(str(p) for p in pages),
            "image_used": os.path.basename(image_used) if image_used else "none",
            "image_score": round(score, 3),
            "seconds": round(seconds, 1),
            "result": "",
            "notes": "",
        })
        print(f"  Q{n:<2} image={rows[-1]['image_used']:<12} "
              f"score={score:.3f}  {seconds:.1f}s")

        prev_q, prev_a = item["q"], answer

    os.makedirs(OUT_DIR, exist_ok=True)
    detail_path = os.path.join(OUT_DIR, f"{config_name}.csv")
    write_csv(detail_path, DETAIL_FIELDS, rows)

    summary = {
        "label": label,
        "k": k,
        "threshold": threshold,
        "questions": len(rows),
        "images_attached": sum(r["image_used"] != "none" for r in rows),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1),
        "errors": sum(r["answer"].startswith("ERROR:") for r in rows),
        "manual_pass": "",
    }
    write_csv(os.path.join(OUT_DIR, "eval_summary.csv"), SUMMARY_FIELDS, [summary], append=True)
    print(f"  saved {detail_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the evaluation question set.")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"text chunks retrieved per question (default {DEFAULT_K})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"image similarity cutoff (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--sweep", action="store_true",
                        help="run the defaults plus one-at-a-time k and threshold variations")
    parser.add_argument("--label", default="",
                        help="tag for runs that used a different index, e.g. chunk800")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not found.")
        print(f"Looked for a .env file at: {chat.ENV_PATH}")
        sys.exit(1)
    if not (os.path.exists(chat.CHROMA_DIR) and os.path.exists(chat.INDEX_STORE_DIR)):
        print("No index found. Run this first: python build_index.py path/to/your.pdf")
        sys.exit(1)

    if args.sweep:
        configs = [(DEFAULT_K, DEFAULT_THRESHOLD)]
        configs += [(k, DEFAULT_THRESHOLD) for k in SWEEP_K_VALUES if k != DEFAULT_K]
        configs += [(DEFAULT_K, t) for t in SWEEP_THRESHOLD_VALUES if t != DEFAULT_THRESHOLD]
    else:
        configs = [(args.k, args.threshold)]

    print(f"{len(configs)} configuration(s) x {len(QUESTIONS)} questions "
          f"= {len(configs) * len(QUESTIONS)} Claude calls")

    print("Loading indexes...")
    vectorstore = chat.load_text_index()
    clip_model, image_embeddings, image_paths = chat.load_image_index()
    try:
        # Sanity check: a count far above what you expect can mean the index
        # holds duplicates or a second document. Re-run build_index.py if so.
        print(f"Text index holds {vectorstore._collection.count()} chunks; "
              f"image index holds {len(image_paths)} pages.")
    except Exception:
        pass

    summaries = [run_config(k, t, args.label, vectorstore, clip_model,
                            image_embeddings, image_paths) for k, t in configs]

    print("\n=== Summary ===")
    print(f"{'k':>3} {'thr':>5} {'images':>7} {'avg s':>6} {'errors':>6}")
    for s in summaries:
        print(f"{s['k']:>3} {s['threshold']:>5.2f} {s['images_attached']:>7} "
              f"{s['avg_seconds']:>6} {s['errors']:>6}")
    print(f"\nOpen the CSVs in ./{OUT_DIR}, fill in the Result column "
          "(Pass / Partial / Fail), and note the totals in manual_pass.")


if __name__ == "__main__":
    main()