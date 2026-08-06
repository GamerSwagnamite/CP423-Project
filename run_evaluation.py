# ----------------------------------------------------------------
# CP423 Project - Evaluation Harness (Retrieval + Generation Metrics)
# ----------------------------------------------------------------
# Name:     Jordan Asmono
# ID:       210922810
# Email:    asmo2810@mylaurier.ca
# Date:     2026-08-05
# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
import json
from pathlib import Path

from corpus_utils import load_corpus
from hybrid_retriever import HybridRetriever
from rag_pipeline import answer_question
# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
CORPUS_DIR = SCRIPT_DIR / "corpus" / "courses"
EVAL_SET_PATH = SCRIPT_DIR / "eval_set.json"
RESULTS_PATH = SCRIPT_DIR / "evaluation_results.json"

RETRIEVAL_TOP_K = 5  # how many chunks the retriever returns per question
# ----------------------------------------------------------------
# Functions
# ----------------------------------------------------------------
def load_eval_set(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_retrieval_metrics(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> dict:
    """Recall@k and MRR against the gold source chunk(s) for one question.

    Unanswerable questions have an empty gold list -- retrieval metrics
    don't really apply to them (there's nothing to "find"), so they're
    skipped for retrieval scoring and evaluated on generation behaviour
    only (did the model correctly say "I don't know").
    """
    if not gold_doc_ids:
        return {"applicable": False}

    gold_set = set(gold_doc_ids)
    retrieved_set = set(retrieved_doc_ids)

    hits = gold_set & retrieved_set
    recall_at_k = len(hits) / len(gold_set)

    # MRR: reciprocal rank of the first gold document found, 0 if none found
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "applicable": True,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "hits": sorted(hits),
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

    eval_questions = load_eval_set(EVAL_SET_PATH)
    print(f"Loaded {len(eval_questions)} evaluation questions")

    results = []
    for q in eval_questions:
        print(f"\nRunning {q['id']} ({q['type']}): {q['question']}")

        result = answer_question(q["question"], retriever, top_k=RETRIEVAL_TOP_K)
        retrieved_doc_ids = [c["doc_id"] for c in result["retrieved_chunks"]]

        retrieval_metrics = compute_retrieval_metrics(retrieved_doc_ids, q["source_doc_ids"])

        print(f"  Retrieved: {retrieved_doc_ids}")
        print(f"  Answer: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")

        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "gold_doc_ids": q["source_doc_ids"],
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_metrics": retrieval_metrics,
            "generated_answer": result["answer"],
            # fill these in manually after reading generated_answer vs reference_answer
            "generation_correct": None,       # true/false
            "generation_cited_correctly": None,  # true/false -- do citations point to real, relevant chunks
            "correctly_abstained": None,       # true/false -- only relevant for unanswerable questions
        })

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # aggregate retrieval metrics across applicable (non-unanswerable) questions
    applicable = [r["retrieval_metrics"] for r in results if r["retrieval_metrics"]["applicable"]]
    if applicable:
        avg_recall = sum(m["recall_at_k"] for m in applicable) / len(applicable)
        avg_mrr = sum(m["reciprocal_rank"] for m in applicable) / len(applicable)
        print(f"\nAggregate retrieval metrics ({len(applicable)} questions with gold chunks):")
        print(f"  Mean Recall@{RETRIEVAL_TOP_K}: {avg_recall:.3f}")
        print(f"  Mean Reciprocal Rank (MRR): {avg_mrr:.3f}")

    print(f"\nSaved full results to {RESULTS_PATH}")
    print("Next: open the file and manually fill in 'generation_correct', "
          "'generation_cited_correctly', and 'correctly_abstained' for each "
          "question by reading the generated answer, then compute your "
          "generation accuracy for the report.")


if __name__ == "__main__":
    main()
