# CP423 - Build and Evaluate a Retrieval-Augmented Generation System 

## Overview:
A hybrid retrieval-augmented generation (RAG) system built over a corpus of University of Waterloo Faculty of Mathematics undergraduate course descriptions. Retrieval combines BM25 (sparse) and BAAI/bge-small-en-v1.5 (dense) via Reciprocal Rank Fusion; generation runs on a locally hosted Qwen2.5 7B-Instruct model served through Ollama.

Full write-up, corpus construction rationale, error analysis, and limitations are in Report.pdf.

## Repository Structure:
```
.
├── README.md
├── requirements.txt
├── Report.pdf                    # full project report
├── build_corpus.py               # fetches + filters the corpus from UW's Open Data API
├── corpus_utils.py                # shared corpus loading helper
├── bm25_retriever.py              # sparse retriever
├── dense_retriever.py             # dense retriever (BAAI/bge-small-en-v1.5)
├── hybrid_retriever.py            # BM25 + dense combined via Reciprocal Rank Fusion
├── rag_pipeline.py                # retrieval + prompt construction + generation
├── run_diagnostic.py              # bare-LLM diagnostic experiment (10 questions, no retrieval)
├── run_evaluation.py              # full evaluation harness (retrieval + generation metrics)
├── reproduce_results.py           # single-command entry point, runs the two scripts above
├── eval_set.json                  # 10 hand-written evaluation questions (factoid/multi-hop/unanswerable)
└── corpus/
    └── courses/                   # 342 course-description JSON documents (see Corpus section below)
```

## Prerequisites:
1. **Python 3.10+** (uses `dict | None` style type hints)
2. **[Ollama](https://ollama.com)** installed and running locally, with the generation model pulled:
   ```
   ollama pull qwen2.5:7b-instruct-q4_K_M
   ```
   Confirm the exact tag with `ollama list` and update `MODEL_NAME` in `run_diagnostic.py` and `rag_pipeline.py` if it differs.
3. A machine that can run a 7B-parameter Q4 quantized model locally (developed and tested on an RTX 3060 12GB / 32GB RAM system; a CPU-only machine will work but generation will be considerably slower).

## Setup
```bash
git clone <this-repo-url>
cd <this-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt --break-system-packages
```

`sentence-transformers` will pull in PyTorch automatically. If you have an NVIDIA GPU and want CUDA acceleration for the embedding model, install the appropriate CUDA build of `torch` from https://pytorch.org/get-started/locally/ before running `pip install -r requirements.txt` (CPU-only PyTorch works fine too, dense retrieval over 342 short documents is fast either way).

## Corpus

The corpus (342 JSON documents in `corpus/courses/`) is included directly in this repository, so no API key is needed to reproduce the reported results as-is.

If you want to regenerate the corpus from scratch (e.g. to pull a different term range or subject list), you'll need a University of Waterloo Open Data API key:

```bash
curl --request POST \
  --url https://openapi.data.uwaterloo.ca/v3/account/register \
  --header 'content-type: application/x-www-form-urlencoded' \
  --data 'email=your@email.com&project=CP423%20RAG%20Project&uri=<this-repo-url>'
```

Then:
```bash
export UWATERLOO_API_KEY=your_key_here     # Windows: set UWATERLOO_API_KEY=your_key_here
python build_corpus.py
```

This fetches course records for Fall 2024, Winter 2025, and Spring 2025 (pinned terms, for reproducibility) across the Faculty of Mathematics subject codes, filters out empty/placeholder descriptions, and writes one JSON file per course to `corpus/courses/`. Data is sourced from the University of Waterloo's official Open Data API and is subject to the University of Waterloo Open Data License.

## Reproducing All Results

With Ollama running and the corpus present, run:

```bash
python reproduce_results.py
```

This is the single command that reproduces every experimental result reported in `Report.pdf`:

1. **Diagnostic experiment** (`run_diagnostic.py`): poses 10 factual questions to bare Qwen2.5 with no retrieved context, saves results to `diagnostic_results.json`.
2. **Full evaluation** (`run_evaluation.py`): runs all 10 hand-written evaluation questions through the complete hybrid RAG pipeline, computes Recall@5 and MRR automatically against known gold source documents, saves results to `evaluation_results.json`.

Both use `temperature=0` and a fixed `seed=42` on every Ollama call for reproducibility.

**Manual grading step:** whether each generated answer is factually correct, whether citations point to the right chunks, and whether unanswerable questions were correctly abstained on are graded by hand (this is disclosed and expected per the assignment; automatic string-matching isn't reliable for free-text answer correctness). After running `reproduce_results.py`, open `diagnostic_results.json` and `evaluation_results.json` and fill in the `null` grading fields (`correct`, `generation_correct`, `generation_cited_correctly`, `correctly_abstained`) by comparing each `model_answer`/`generated_answer` against its `ground_truth_answer`/`reference_answer`. Re-running `reproduce_results.py` afterward will print the aggregate accuracy once those fields are filled in.

## Running Individual Components

Each script can also be run on its own for inspection/debugging:

```bash
python bm25_retriever.py       # demo BM25 query
python dense_retriever.py      # demo dense retrieval query (embeds + caches corpus on first run)
python hybrid_retriever.py     # demo fused BM25 + dense query
python rag_pipeline.py         # demo full retrieve-then-generate pipeline on one question
```

Dense embeddings are cached to `corpus_embeddings.npy` with a companion `corpus_embeddings_manifest.json` after the first run; the manifest is checked against the current corpus's document IDs on every load, so a stale cache is automatically detected and recomputed rather than silently reused.

## Random Seeds

Generation uses a fixed seed (`42`) and `temperature=0` throughout (`run_diagnostic.py`, `rag_pipeline.py`) for reproducible outputs. Retrieval components (BM25, dense embeddings) are fully deterministic given a fixed corpus and model, no seed is needed there.

## License / Data Attribution

Course description data is sourced from the University of Waterloo Open Data API (https://openapi.data.uwaterloo.ca) and is subject to the University of Waterloo Open Data License. See https://uwaterloo.ca/api/ for details.
