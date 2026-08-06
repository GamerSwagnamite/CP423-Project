# ----------------------------------------------------------------
# CP423 Project - Get API Key
# ----------------------------------------------------------------
# Name:     Jordan Asmono
# ID:       210922810
# Email:    asmo2810@mylaurier.ca
# Date:     2026-08-05
# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
import json
import os
import re
import time
from pathlib import Path
import requests
# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
API_KEY = os.environ.get("UWATERLOO_API_KEY", "68273326D21A47EB95F8B0703F7485A8")
BASE_URL = "https://openapi.data.uwaterloo.ca/v3"
COURSES_BY_SUBJECT_PATH = "/Courses/{term_code}/{subject}"

# Subjects housed in the Faculty of Mathematics (adjust as needed)
MATH_FACULTY_SUBJECTS = [
    "CS", "MATH", "STAT", "AMATH", "CO", "PMATH",
    "ACTSC", "SE", "BIOSTAT", "CM",
]

# pin FIXED past terms for reproducibility -- one academic year, all three
# terms, so courses offered in any single term are still captured.
# term code format CYYM: C=century(2000s=1), YY=year, M=month(Fall=9,Winter=1,Spring=5)
TERM_CODES = [
    "1249",  # Fall 2024
    "1251",  # Winter 2025
    "1255",  # Spring 2025
]

# which academic career(s) to keep. the API mixes undergrad and grad courses.
ACADEMIC_CAREERS_TO_KEEP = {"UG"}  # add "GRD" here too if you want grad courses

# stub/placeholder descriptions provide nothing for retrieval to find, and
# can quietly hurt precision by embedding ambiguously close to real content.
# filtered out alongside empty descriptions below.
MIN_DESCRIPTION_WORDS = 8
STUB_PATTERNS = [
    r"refer to .* calendar",
    r"see .* calendar",
    r"taught at .*university",
    r"cross[- ]listed",
]

OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "corpus" / "courses"
REQUEST_DELAY_SECONDS = 1.5   # be polite / stay under the per-minute rate limit
MAX_RETRIES = 3
DEBUG = False
# ----------------------------------------------------------------
# Functions
# ----------------------------------------------------------------
def api_get(path: str) -> dict:
    """GET a path from the API, handling the x-api-key header and 429 backoff."""
    url = f"{BASE_URL}{path}"
    headers = {"x-api-key": API_KEY}

    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(url, headers=headers, timeout=30)

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 429:
            wait = 5 * attempt
            print(f"  Rate limited on {url}, waiting {wait}s (attempt {attempt})")
            time.sleep(wait)
            continue

        print(f"  ERROR {resp.status_code} fetching {url}: {resp.text[:200]}")
        return {}

    print(f"  Giving up on {url} after {MAX_RETRIES} attempts")
    return {}


def fetch_courses_for_subject_term(subject: str, term_code: str) -> list:
    data = api_get(COURSES_BY_SUBJECT_PATH.format(term_code=term_code, subject=subject))
    if isinstance(data, dict):
        # adjust this if the live response wraps the list differently,
        # e.g. {"data": [...]} -- print(data) once to confirm the shape
        return data.get("data", []) if "data" in data else (data if isinstance(data, list) else [])
    if isinstance(data, list):
        return data
    return []


def merge_course_records(existing: dict, new: dict) -> dict:
    """Prefer whichever record actually has a non-empty description."""
    if not existing.get("description") and new.get("description"):
        return new
    return existing


def is_stub_description(description: str) -> bool:
    """Flag near-empty or placeholder descriptions (e.g. cross-listed courses
    that just point to another institution's calendar) that give retrieval
    nothing real to find."""
    if len(description.split()) < MIN_DESCRIPTION_WORDS:
        return True

    lowered = description.lower()
    for pattern in STUB_PATTERNS:
        if re.search(pattern, lowered):
            return True

    return False


def build_document(course: dict) -> dict | None:
    subject = course.get("subjectCode", "")
    catalog_number = course.get("catalogNumber", "")
    title = course.get("title", "")
    description = (course.get("description") or "").strip()
    requirements = (course.get("requirementsDescription") or "").strip()
    course_id = course.get("courseId") or f"{subject}_{catalog_number}"
    career = course.get("associatedAcademicCareer", "")

    if not description:
        return None  # nothing for retrieval to find

    if is_stub_description(description):
        return None  # placeholder/cross-listing stub, not real content

    text_parts = [
        f"{subject} {catalog_number}: {title}",
        f"Description: {description}",
    ]
    if requirements:
        text_parts.append(f"Requirements: {requirements}")

    return {
        "doc_id": f"uw-course-{subject}-{catalog_number}",
        "source": "University of Waterloo Open Data API (Courses endpoint)",
        "subject": subject,
        "catalog_number": catalog_number,
        "title": title,
        "academic_career": career,
        "requirements": requirements,
        "text": "\n".join(text_parts),
    }

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        raise SystemExit(
            "Set your API key: export UWATERLOO_API_KEY=your_key_here, "
            "or edit API_KEY at the top of this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subject in MATH_FACULTY_SUBJECTS:
        print(f"Fetching {subject}...")
        merged: dict[str, dict] = {}  # courseId -> best course record seen

        for term_code in TERM_CODES:
            courses = fetch_courses_for_subject_term(subject, term_code)
            print(f"  {term_code}: {len(courses)} raw records fetched")

            if courses and DEBUG:
                sample = courses[0]
                print(f"    sample record keys: {list(sample.keys())}")
                print(f"    sample career value: {sample.get('associatedAcademicCareer')!r}")
                print(f"    sample description present: {bool((sample.get('description') or '').strip())}")

            careers_seen = set()
            kept_this_term = 0
            for course in courses:
                career = course.get("associatedAcademicCareer", "")
                careers_seen.add(career)
                if career not in ACADEMIC_CAREERS_TO_KEEP:
                    continue
                cid = course.get("courseId")
                if cid is None:
                    continue
                kept_this_term += 1
                if cid in merged:
                    merged[cid] = merge_course_records(merged[cid], course)
                else:
                    merged[cid] = course

            print(f"    careers seen this term: {careers_seen}")
            print(f"    kept after career filter: {kept_this_term}")

            time.sleep(REQUEST_DELAY_SECONDS)

        no_description_count = 0
        stub_count = 0
        saved = 0
        for course in merged.values():
            description = (course.get("description") or "").strip()
            if not description:
                no_description_count += 1
                continue
            if is_stub_description(description):
                stub_count += 1
                continue

            doc = build_document(course)
            if doc is None:
                continue

            out_path = OUTPUT_DIR / f"{doc['doc_id']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            saved += 1

        print(f"  -> {saved} documents saved for {subject} "
              f"({len(merged)} unique courses after career filter, "
              f"{no_description_count} dropped for missing description, "
              f"{stub_count} dropped as stub/placeholder descriptions)")

    all_docs = list(OUTPUT_DIR.glob("*.json"))
    print(f"\nDone. Total documents in corpus: {len(all_docs)}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()