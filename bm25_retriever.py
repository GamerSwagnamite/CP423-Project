# ----------------------------------------------------------------
# CP423 Project - BM25 Retriever
# ----------------------------------------------------------------
# Name:     Jordan Asmono
# ID:       210922810
# Email:    asmo2810@mylaurier.ca
# Date:     2026-08-05
# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from corpus_utils import load_corpus
# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
CORPUS_DIR = SCRIPT_DIR / "corpus" / "courses"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "for", "on", "with", "at", "by", "from", "up",
    "about", "into", "through", "during", "and", "or", "but", "if",
    "then", "else", "when", "what", "which", "who", "whom", "this",
    "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "do", "does", "did", "will", "would", "should", "can",
    "could", "has", "have", "had", "as", "not", "no", "so",
}

# light domain-specific normalization: the corpus uses abbreviations
# (Prereq, Antireq, Coreq) that a natural-language question is unlikely to
# use verbatim ("prerequisite"). Mapping these closes an easily-avoidable
# vocabulary gap. Disclose this preprocessing choice in the report.
NORMALIZE_MAP = {
    "prerequisite": "prereq",
    "prerequisites": "prereq",
    "antirequisite": "antireq",
    "antirequisites": "antireq",
    "corequisite": "coreq",
    "corequisites": "coreq",
}
# ----------------------------------------------------------------
# Functions
# ----------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, normalize domain terms, drop stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [NORMALIZE_MAP.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class BM25Retriever:
    def __init__(self, documents: list[dict]):
        """documents: list of dicts, each must have a 'text' field and a 'doc_id' field."""
        self.documents = documents
        self.tokenized_corpus = [tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k documents with their BM25 scores, highest first."""
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "doc_id": self.documents[idx]["doc_id"],
                "score": float(scores[idx]),
                "text": self.documents[idx]["text"],
                "subject": self.documents[idx].get("subject"),
                "catalog_number": self.documents[idx].get("catalog_number"),
            })
        return results
# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print(f"Loading corpus from {CORPUS_DIR}...")
    documents = load_corpus(CORPUS_DIR)
    print(f"Loaded {len(documents)} documents")

    retriever = BM25Retriever(documents)

    demo_query = "What is the prerequisite for STAT 442?"
    print(f"\nDemo query: {demo_query!r}")
    results = retriever.search(demo_query, top_k=5)

    for rank, r in enumerate(results, start=1):
        print(f"\n{rank}. [{r['doc_id']}] score={r['score']:.3f}")
        print(f"   {r['text'][:150]}...")


if __name__ == "__main__":
    main()