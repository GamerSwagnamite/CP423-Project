# ----------------------------------------------------------------
# CP423 Project - Diagnostic Experiment (Bare LLM Baseline)
# ----------------------------------------------------------------
# Name:     Jordan Asmono
# ID:       210922810
# Email:    asmo2810@mylaurier.ca
# Date:     2026-08-05
# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
import json
import time
from pathlib import Path

import requests
# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"  # match whatever you pulled in Ollama

SCRIPT_DIR = Path(__file__).parent
OUTPUT_PATH = SCRIPT_DIR / "diagnostic_results.json"

# 10 factual questions whose answers exist in the corpus, with ground truth
# and source doc IDs so results can be graded and cited in the report.
DIAGNOSTIC_QUESTIONS = [
    {
        "id": "q1",
        "question": "At the University of Waterloo, what is the prerequisite for STAT 442 (Data Visualization)?",
        "ground_truth_answer": "STAT 341",
        "source_doc_id": "uw-course-STAT-442",
    },
    {
        "id": "q2",
        "question": "What are the two prerequisite courses for ACTSC 454 (Longevity and Mortality Using Predictive Analytics), and which students is the course restricted to?",
        "ground_truth_answer": "ACTSC 331 and STAT 330; restricted to Actuarial Science or Mathematical Finance students only",
        "source_doc_id": "uw-course-ACTSC-454",
    },
    {
        "id": "q3",
        "question": "What are the antirequisites for AMATH 331 (Applied Real Analysis) at the University of Waterloo?",
        "ground_truth_answer": "PMATH 333 and PMATH 351",
        "source_doc_id": "uw-course-AMATH-331",
    },
    {
        "id": "q4",
        "question": "What are the antirequisites for CO 227 (Introduction to Optimization, Non-Specialist Level)?",
        "ground_truth_answer": "CO 250, CO 255, CO 352",
        "source_doc_id": "uw-course-CO-227",
    },
    {
        "id": "q5",
        "question": "Which computational complexity classes are explicitly mentioned in the CO 454 (Scheduling) course description?",
        "ground_truth_answer": "P, NP, NP-complete, and NP-hard",
        "source_doc_id": "uw-course-CO-454",
    },
    {
        "id": "q6",
        "question": "At the University of Waterloo, what minimum grade in CS 115 qualifies a student for one of the alternative prerequisite paths into CS 136?",
        "ground_truth_answer": "At least 90% in CS 115",
        "source_doc_id": "uw-course-CS-136",
    },
    {
        "id": "q7",
        "question": "What minimum grade in CS 136 or CS 146 is required to enroll in CS 251E (Computer Organization and Design, Enriched)?",
        "ground_truth_answer": "85% or higher",
        "source_doc_id": "uw-course-CS-251E",
    },
    {
        "id": "q8",
        "question": "Besides CS 338, which ECE courses are listed as antirequisites for CS 348 (Introduction to Database Management)?",
        "ground_truth_answer": "ECE 356 and ECE 456",
        "source_doc_id": "uw-course-CS-348",
    },
    {
        "id": "q9",
        "question": "What is the antirequisite course for CS 447 (Software Testing, Quality Assurance, and Maintenance)?",
        "ground_truth_answer": "SE 465",
        "source_doc_id": "uw-course-CS-447",
    },
    {
        "id": "q10",
        "question": "What is the antirequisite course for MATH 247 (Calculus 3, Advanced Level)?",
        "ground_truth_answer": "MATH 237",
        "source_doc_id": "uw-course-MATH-247",
    },
]
# ----------------------------------------------------------------
# Functions
# ----------------------------------------------------------------
def ask_bare_llm(question: str) -> str:
    """Send a question to Qwen2.5 via Ollama with NO retrieved context."""
    prompt = (
        "Answer the following question as accurately and specifically as "
        "you can. If you don't know the answer, say so explicitly rather "
        "than guessing.\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()
# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    results = []
    for q in DIAGNOSTIC_QUESTIONS:
        print(f"Asking {q['id']}: {q['question']}")
        answer = ask_bare_llm(q["question"])
        print(f"  -> {answer[:150]}{'...' if len(answer) > 150 else ''}")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth_answer": q["ground_truth_answer"],
            "source_doc_id": q["source_doc_id"],
            "model_answer": answer,
            "correct": None,  # fill in manually after reading the answer
        })
        time.sleep(0.5)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")
    print("Next: open the file, fill in 'correct': true/false for each by "
          "comparing model_answer to ground_truth_answer, then compute your "
          "accuracy for the report.")


if __name__ == "__main__":
    main()