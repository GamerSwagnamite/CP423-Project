# ----------------------------------------------------------------
# CP423 Project - Hybrid Retriever (BM25 + Dense, via RRF)
# ----------------------------------------------------------------
# Name:     Jordan Asmono
# ID:       210922810
# Email:    asmo2810@mylaurier.ca
# Date:     2026-08-05
# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
from pathlib import Path

from bm25_retriever import BM25Retriever
from corpus_utils import load_corpus
from dense_retriever import DenseRetriever
# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
CORPUS_DIR = SCRIPT_DIR / "corpus" / "courses"

# how many candidates each retriever contributes to the fusion pool before
# re-ranking. Larger than the final top_k so RRF has enough to work with.
CANDIDATE_POOL_SIZE = 20

# RRF constant -- 60 is the standard default from the original RRF paper
# (Cormack et al.), chosen to de-emphasize the exact rank position of any
# single retriever and keep the fusion robust to outlier rankings.
RRF_K = 60
# ----------------------------------------------------------------
# Functions
# ----------------------------------------------------------------
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Combine multiple ranked lists of doc_ids into one fused score per doc_id.

    score(d) = sum over lists of 1 / (k + rank_in_that_list)
    Only lists a document appears in contribute -- absence from a list
    simply means zero contribution from that retriever, not a penalty.
    """
    fused_scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused_scores


class HybridRetriever:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.doc_lookup = {doc["doc_id"]: doc for doc in documents}

        print("Building BM25 index...")
        self.bm25_retriever = BM25Retriever(documents)

        print("Building dense index...")
        self.dense_retriever = DenseRetriever(documents)

    def search(self, query: str, top_k: int = 5, debug: bool = False) -> list[dict]:
        bm25_results = self.bm25_retriever.search(query, top_k=CANDIDATE_POOL_SIZE)
        dense_results = self.dense_retriever.search(query, top_k=CANDIDATE_POOL_SIZE)

        bm25_ids = [r["doc_id"] for r in bm25_results]
        dense_ids = [r["doc_id"] for r in dense_results]

        if debug:
            print(f"  [debug] query: {query!r}")
            print(f"  [debug] bm25 candidate pool ({len(bm25_ids)}): {bm25_ids}")
            print(f"  [debug] dense candidate pool ({len(dense_ids)}): {dense_ids}")

        fused_scores = reciprocal_rank_fusion([bm25_ids, dense_ids])
        ranked_doc_ids = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)[:top_k]

        # keep per-retriever scores around for transparency/debugging/report tables
        bm25_score_by_id = {r["doc_id"]: r["score"] for r in bm25_results}
        dense_score_by_id = {r["doc_id"]: r["score"] for r in dense_results}

        results = []
        for doc_id in ranked_doc_ids:
            doc = self.doc_lookup[doc_id]
            results.append({
                "doc_id": doc_id,
                "rrf_score": fused_scores[doc_id],
                "bm25_score": bm25_score_by_id.get(doc_id),
                "dense_score": dense_score_by_id.get(doc_id),
                "text": doc["text"],
                "subject": doc.get("subject"),
                "catalog_number": doc.get("catalog_number"),
            })
        return results
# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print(f"Loading corpus from {CORPUS_DIR}...")
    documents = load_corpus(CORPUS_DIR)
    print(f"Loaded {len(documents)} documents")

    retriever = HybridRetriever(documents)

    demo_query = "What is the prerequisite for STAT 442?"
    print(f"\nDemo query: {demo_query!r}")
    results = retriever.search(demo_query, top_k=5, debug=True)

    for rank, r in enumerate(results, start=1):
        bm25_str = f"{r['bm25_score']:.3f}" if r['bm25_score'] is not None else "n/a"
        dense_str = f"{r['dense_score']:.3f}" if r['dense_score'] is not None else "n/a"
        print(f"\n{rank}. [{r['doc_id']}] rrf={r['rrf_score']:.5f} "
              f"(bm25={bm25_str}, dense={dense_str})")
        print(f"   {r['text'][:150]}...")


if __name__ == "__main__":
    main()