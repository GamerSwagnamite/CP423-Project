# ----------------------------------------------------------------
# CP423 Project - Reproduce All Results
# ----------------------------------------------------------------
# Name:     Jordan Asmono
# ID:       210922810
# Email:    asmo2810@mylaurier.ca
# Date:     2026-08-05
# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
import json
import subprocess
import sys
from pathlib import Path
# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DIAGNOSTIC_RESULTS_PATH = SCRIPT_DIR / "diagnostic_results.json"
EVALUATION_RESULTS_PATH = SCRIPT_DIR / "evaluation_results.json"
# ----------------------------------------------------------------
# Functions
# ----------------------------------------------------------------
def run_step(description: str, script_name: str):
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    result = subprocess.run([sys.executable, str(SCRIPT_DIR / script_name)])
    if result.returncode != 0:
        raise SystemExit(f"{script_name} exited with code {result.returncode}, stopping.")


def print_summary():
    print(f"\n{'=' * 60}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'=' * 60}")

    if DIAGNOSTIC_RESULTS_PATH.exists():
        with open(DIAGNOSTIC_RESULTS_PATH, "r", encoding="utf-8") as f:
            diagnostic = json.load(f)
        graded = [r for r in diagnostic if r.get("correct") is not None]
        if graded:
            correct = sum(1 for r in graded if r["correct"])
            print(f"Diagnostic (bare LLM) accuracy: {correct}/{len(graded)}")
        else:
            print(f"Diagnostic results saved to {DIAGNOSTIC_RESULTS_PATH.name} "
                  f"-- 'correct' fields still need manual grading.")

    if EVALUATION_RESULTS_PATH.exists():
        with open(EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
            evaluation = json.load(f)

        applicable = [r["retrieval_metrics"] for r in evaluation if r["retrieval_metrics"]["applicable"]]
        if applicable:
            avg_recall = sum(m["recall_at_k"] for m in applicable) / len(applicable)
            avg_mrr = sum(m["reciprocal_rank"] for m in applicable) / len(applicable)
            print(f"Mean Recall@5: {avg_recall:.3f}")
            print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.3f}")

        graded = [r for r in evaluation if r.get("generation_correct") is not None]
        if graded:
            correct = sum(1 for r in graded if r["generation_correct"])
            print(f"Generation accuracy: {correct}/{len(graded)}")
        else:
            print(f"Evaluation results saved to {EVALUATION_RESULTS_PATH.name} "
                  f"-- generation grading fields still need manual completion.")
# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print("Reproducing all experimental results for the CP423 RAG project.")
    print("Requires: Ollama running locally with qwen2.5:7b-instruct-q4_K_M pulled, "
          "and corpus/courses/ populated (see README for corpus setup).")

    run_step("STEP 1: Diagnostic experiment (bare LLM baseline)", "run_diagnostic.py")
    run_step("STEP 2: Retrieval + generation evaluation (hybrid RAG pipeline)", "run_evaluation.py")

    print_summary()

    print("\nDone. See diagnostic_results.json and evaluation_results.json for full "
          "per-question detail. Manual grading fields ('correct', 'generation_correct', "
          "'generation_cited_correctly', 'correctly_abstained') must be filled in by "
          "hand before the summary above is complete -- see README.md.")


if __name__ == "__main__":
    main()
