# ----------------------------------------------------------------
# CP423 Project - RAG Pipeline (Retrieval + Generation)
# ----------------------------------------------------------------
# Name:     Jordan Asmono
# ID:       210922810
# Email:    asmo2810@mylaurier.ca
# Date:     2026-08-05
# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
from pathlib import Path

import requests

from corpus_utils import load_corpus
from hybrid_retriever import HybridRetriever
# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
CORPUS_DIR = SCRIPT_DIR / "corpus" / "courses"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"  # match whatever you pulled in Ollama

# fixed seed + zero temperature for reproducible outputs across runs --
# without this, Ollama samples stochastically and re-running the same
# question can produce a different (and differently-graded) answer.
RANDOM_SEED = 42

RAG_TOP_K = 5  # how many retrieved chunks to give the LLM as context

SYSTEM_INSTRUCTIONS = """You are a course advising assistant for the University of Waterloo Faculty of Mathematics. Answer the user's question using ONLY the information in the numbered context chunks below.

Rules you must follow:
1. Base your answer strictly on the provided context. Do not use outside knowledge.
2. Every factual claim in your answer must include an inline citation to the chunk(s) it came from, using the format [1], [2], etc.
3. If the context does not contain enough information to answer the question, respond with exactly: "I don't know" followed by a brief explanation of what's missing. Do not guess or fill gaps with outside knowledge.
4. Be concise and direct."""
# ----------------------------------------------------------------
# Functions
# ----------------------------------------------------------------
def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Assemble the numbered context block + question into a single prompt."""
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_lines.append(f"[{i}] (doc_id: {chunk['doc_id']})\n{chunk['text']}")
    context_block = "\n\n".join(context_lines)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer (with inline citations like [1]):"
    )


def call_llm(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"seed": RANDOM_SEED, "temperature": 0},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def answer_question(query: str, retriever: HybridRetriever, top_k: int = RAG_TOP_K) -> dict:
    """Run the full retrieve -> prompt -> generate pipeline for one question."""
    retrieved_chunks = retriever.search(query, top_k=top_k)
    prompt = build_prompt(query, retrieved_chunks)
    answer = call_llm(prompt)

    return {
        "query": query,
        "retrieved_chunks": [
            {"doc_id": c["doc_id"], "rrf_score": c["rrf_score"]} for c in retrieved_chunks
        ],
        "prompt": prompt,
        "answer": answer,
    }
# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print(f"Loading corpus from {CORPUS_DIR}...")
    documents = load_corpus(CORPUS_DIR)
    print(f"Loaded {len(documents)} documents")

    print("Building hybrid retriever...")
    retriever = HybridRetriever(documents)

    demo_query = "What is the prerequisite for STAT 442?"
    print(f"\nQuestion: {demo_query}")

    result = answer_question(demo_query, retriever)

    print("\nRetrieved chunks:")
    for i, c in enumerate(result["retrieved_chunks"], start=1):
        print(f"  [{i}] {c['doc_id']} (rrf={c['rrf_score']:.5f})")

    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()