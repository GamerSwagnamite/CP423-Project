# ----------------------------------------------------------------
# CP423 Project - Dense Retriever
# ----------------------------------------------------------------
# Name:     Jordan Asmono
# ID:       210922810
# Email:    asmo2810@mylaurier.ca
# Date:     2026-08-05
# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from corpus_utils import load_corpus
# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
CORPUS_DIR = SCRIPT_DIR / "corpus" / "courses"
EMBEDDINGS_CACHE_PATH = SCRIPT_DIR / "corpus_embeddings.npy"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models are trained asymmetrically: queries get this instruction
# prefix, documents/passages do NOT. Getting this backwards quietly hurts
# retrieval quality without throwing any errors, so keep it front and center.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# ----------------------------------------------------------------
# Functions
# ----------------------------------------------------------------
class DenseRetriever:
    def __init__(self, documents: list[dict], use_cache: bool = True):
        """documents: list of dicts, each must have 'text' and 'doc_id' fields."""
        self.documents = documents
        self.model = SentenceTransformer(MODEL_NAME)

        if use_cache and EMBEDDINGS_CACHE_PATH.exists():
            print(f"Loading cached embeddings from {EMBEDDINGS_CACHE_PATH}")
            self.doc_embeddings = np.load(EMBEDDINGS_CACHE_PATH)
            if self.doc_embeddings.shape[0] != len(documents):
                print("  Cache size mismatch with corpus -- recomputing.")
                self.doc_embeddings = self._embed_documents()
        else:
            self.doc_embeddings = self._embed_documents()

    def _embed_documents(self) -> np.ndarray:
        print(f"Embedding {len(self.documents)} documents with {MODEL_NAME}...")
        texts = [doc["text"] for doc in self.documents]
        # no instruction prefix for documents/passages -- only queries get one.
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )
        np.save(EMBEDDINGS_CACHE_PATH, embeddings)
        print(f"Cached embeddings to {EMBEDDINGS_CACHE_PATH}")
        return embeddings

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k documents by cosine similarity, highest first.

        Embeddings are normalized, so dot product == cosine similarity.
        """
        query_with_instruction = QUERY_INSTRUCTION + query
        query_embedding = self.model.encode(
            query_with_instruction, normalize_embeddings=True
        )

        scores = self.doc_embeddings @ query_embedding  # (N,) dot products

        ranked_indices = np.argsort(-scores)[:top_k]

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

    retriever = DenseRetriever(documents)

    demo_query = "What is the prerequisite for STAT 442?"
    print(f"\nDemo query: {demo_query!r}")
    results = retriever.search(demo_query, top_k=5)

    for rank, r in enumerate(results, start=1):
        print(f"\n{rank}. [{r['doc_id']}] score={r['score']:.3f}")
        print(f"   {r['text'][:150]}...")


if __name__ == "__main__":
    main()