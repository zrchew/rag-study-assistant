# RAG Study Assistant

A retrieval-augmented generation (RAG) pipeline that answers questions from PDF documents, combining text and image retrieval so it can use tables, charts and figures that plain text extraction misses.

Ask it a question, and it:
1. Retrieves the most relevant text chunks from the document (via a local sentence-embedding model + Chroma)
2. Separately checks whether any single page's rendered image is a strong enough match to be worth attaching (via CLIP)
3. Sends the text context, and the image if one qualified, to Claude, which answers only from what it was given

## Why multimodal retrieval

Many real PDFs — slide decks, papers with dense tables and figures — put a meaningful share of their content in images, not extractable text. A text-only RAG pipeline is blind to that content. This project adds a second retrieval path so figures and tables can be used, without attaching every page's image to every question.

## Architecture

```
build_index.py  (run once per document)
  ├─ PDF → text chunks → HuggingFace embeddings → Chroma  (./chroma_db)
  └─ PDF → page images → CLIP embeddings           (./index_store)

chat.py  (run repeatedly, no re-indexing)
  ├─ question → text search (Chroma)         → top-k chunks
  ├─ question → CLIP text embedding           → best-matching page image, if above threshold
  └─ [text chunks] + [image, if any] → Claude → answer, with conversation memory for follow-ups
```

Indexing and querying are deliberately separate: embedding a document (especially the CLIP image pass) is the expensive step, so it happens once and is reused across every question afterward.

**Image relevance, not image-per-question.** Rather than attaching every page image to every call, the question is compared against every page's CLIP embedding, and only the best match — if it clears a similarity threshold — gets attached. This keeps images out of purely textual questions and limits the multimodal cost to the questions that actually need it.

## Tech stack

| Component | Tool |
|---|---|
| Text embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Image/text shared embeddings | `clip-ViT-B-32` (local, free) |
| Vector store | Chroma (persisted to disk) |
| PDF → page images | PyMuPDF (`fitz`) |
| Generation | Claude, via `langchain-anthropic` |
| Orchestration | LangChain |

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root (never committed — see `.gitignore`):
```
ANTHROPIC_API_KEY=your_key_here
```

Put a PDF in `data/`, then build its index once:
```powershell
python build_index.py data/your-file.pdf
```

Then chat with it, as many times as you like, without re-indexing:
```powershell
python chat.py
```

## Evaluation

`run_eval.py` runs a fixed set of 11 questions through the pipeline and saves the answers to CSV for manual grading (Pass / Partial / Fail). It supports overriding `k` (chunks retrieved) and the image similarity threshold from the command line, so different configurations can be compared without editing code:

```powershell
python run_eval.py                    # defaults: k=6, threshold=0.22
python run_eval.py --k 10             # more context per question
python run_eval.py --threshold 0.30   # stricter image relevance
```

`diagnose.py` is a free, no-API-call debugging tool: given a question and a phrase you know is in the correct answer, it ranks every chunk in the index and reports exactly where that phrase's chunk falls in the ranking. It answers "did retrieval miss it, or did the model miss it?" without spending anything.

```powershell
python diagnose.py "What does SCALE stand for?" "Context-Aware Learning Engine"
```

### Test document

This evaluation was run against an arXiv paper on [DeepEdu-v1: Efficient and Scalable Agentic LLMs for Vietnamese EducationAI tutoring systems] (https://arxiv.org/abs/2609.31568) rather than the course material the pipeline was originally developed against, for two reasons: its content post-dates Claude's training data (so answers can only come from retrieval, not memorized knowledge), and its license permits redistribution, unlike course slides. The PDF is not included in this repo; download it and place it in `data/` to reproduce these results.

## Results

One document, 11 questions (7 plain text, 2 tables, 1 figure-only value, 1 follow-up, 1 designed to require declining an answer), graded by hand, one run per configuration.

| Configuration | Pass | Partial | Fail | Images attached | Avg. time/question |
|---|---|---|---|---|---|
| k=6, threshold=0.22 (baseline) | 8 | 1 | 2 | 11/11 | 4.2s |
| k=10, threshold=0.22 | 8 | 1 | 2 | 11/11 | 4.6s |
| k=3, threshold=0.22 | 5 | 0 | 6 | 11/11 | 4.3s |
| k=6, threshold=0.30 (images off) | 6 | 1 | 4 | 0/11 | 3.2s |

### What changing `k` showed

Raising `k` from 6 to 10 changed nothing: the same two questions failed, for the same reason (see below). Lowering it to 3 was worse, and not just numerically — two previously-correct answers (the exact retrieval-call reduction, a table lookup) fell out of the top-3 and the model correctly declined rather than guessing. One question (a follow-up, "what about SCR with Lmax = 4096?") did not decline: it answered confidently using an unrelated table, a genuine hallucination rather than a refusal. This was the only unhedged wrong answer across all four configurations, and it only appeared once retrieval had degraded enough (`k=3`) and the question was a follow-up, whose retrieval search uses only its own wording, not the conversation history.

### What removing images showed

At threshold=0.30, no page's image ever scored high enough to be attached, giving a clean text-only comparison against the same `k=6` baseline. Two questions got worse. One (a figure-only numeric value) was already failing at the lower threshold, for a different reason — see below. The other newly failed: the model hedged and then guessed a wrong acronym expansion, on a question that had been answered correctly, from identical retrieved text, when an image was attached. The attached image may have made a passage more legible than the raw extracted text, on a page where the extraction was imperfect.

### Two failures that no retrieval setting fixed

Two questions failed identically across every text-retrieval setting tested (k=3, 6, 10):

- **"What does SCALE stand for?"** — the chunk containing the definition ranks **#64 of 112** chunks for this question.
- **"Did the paper test on real Vietnamese textbooks?"** — the chunk containing the relevant statement ranks **#18 of 112**.

Neither rank is reachable by any `k` worth using in practice. The initial hypothesis was that this was a **truncation problem**: `all-MiniLM-L6-v2`'s embeddings only consider the first 256 tokens of a chunk (confirmed directly from the model's own configuration), so a 1200-character chunk (roughly 250-350 tokens for this dense, symbol-heavy text) could plausibly be losing its ending. This was tested directly: rebuilding the index at `chunk_size=600` (well under the token limit) made both ranks *worse* (#64→#101, #18→#27 out of a now-larger 215-chunk index), which rules out truncation as the primary cause.

The more likely explanation: these are both questions asking for one *specific fact* inside a document where many other chunks discuss the *same topic* at length without stating that fact. The embedding model appears to rank by topical similarity, so chunks that repeat the surrounding subject (e.g. many mentions of "SCALE" describing its components) outrank the one chunk that actually defines the term. In both cases, the model correctly declined to answer rather than guessing — the retrieval failed, but the generation step did not compound it into a wrong answer.

A promising untested next step for this specific failure mode is **query rewriting**: searching with an expanded version of the question (e.g. adding "full name", "definition") rather than the user's literal wording, which is a standard technique for closing exactly this kind of semantic gap.

### The image-retrieval failure

One question ("what is the TPOT value in Figure 1 panel (c)?") failed for an entirely separate reason: CLIP's page-image retrieval did not reliably select the page the figure is actually on. The value exists only as pixels in a chart, in this specific PDF's page 9. Text retrieval did find page 9 (via its caption), but the CLIP similarity score for page 9 was not the highest among all pages for this question, so a different, unrelated page's image was attached instead — or no image was attached at all, depending on the threshold. Raising `k` cannot fix this, since `k` only governs text retrieval; the failure is entirely in how the image-similarity model ranks pages against a question, not in how much text context the model receives.

## Limitations of this evaluation

This is a case study, not a benchmark. One document, 11 questions, one grader, one run per configuration. With this few questions, one flipped answer moves the pass count by roughly 9 percentage points, so small differences between configurations shouldn't be read as conclusive. The settings that worked best here (k=6, threshold≈0.22) are specific to this document's structure and are not claimed to generalize — a different PDF (denser images, shorter pages, a different topic) would likely need retuning. The evaluation harness (`run_eval.py`, `diagnose.py`) is built to be reusable: point it at a different index and question set to test that.

## Repository structure

```
build_index.py     one-time indexing: text + image embeddings, saved to disk
chat.py            interactive chat loop, loads the saved index, supports follow-ups
diagnose.py        free retrieval debugger — ranks all chunks for a question/phrase
run_eval.py        evaluation harness — fixed question set, configurable k/threshold
data/              put PDFs here (not committed)
eval_results/      graded evaluation CSVs
```
