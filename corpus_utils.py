# ----------------------------------------------------------------
# CP423 Project - Corpus Utilities
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
# ----------------------------------------------------------------
# Functions
# ----------------------------------------------------------------
def load_corpus(corpus_dir: Path) -> list[dict]:
    """Load every document JSON file into a list of dicts, sorted by filename
    for a stable, reproducible ordering (important since we cache embeddings
    by index position)."""
    docs = []
    for path in sorted(corpus_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
            docs.append(doc)
    return docs