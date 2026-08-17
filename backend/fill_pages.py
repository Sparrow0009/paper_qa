"""Helper for filling in expected_pages in the eval set.

Reads the PDF directly and suggests which pages are likely to contain each
expected answer, by scoring pages on how many distinctive terms from the
expected_answer they contain.

This deliberately does NOT use the retrieval system. Building the answer key
from the system's own output would mean grading the system against itself, and
it would score close to 100% while measuring nothing.

Treat the output as suggestions. Open the PDF, check, then write the numbers in.

Usage:
    python fill_pages.py paper.pdf
    python fill_pages.py paper.pdf --write     # writes best guesses into the file
"""

import argparse
import json
import re
from collections import Counter

from pypdf import PdfReader

STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "these", "those", "are",
    "was", "were", "which", "their", "them", "they", "have", "has", "had", "not",
    "but", "can", "may", "such", "than", "then", "when", "what", "who", "how",
    "its", "it's", "into", "onto", "over", "under", "about", "also", "other",
    "others", "more", "most", "some", "any", "all", "each", "both", "does",
    "did", "doing", "done", "being", "been", "because", "while", "where",
    "there", "here", "your", "you", "our", "his", "her", "she", "him", "one",
    "two", "three", "four", "five", "paper", "survey", "document", "authors",
    "author", "study", "studies", "work", "works", "section", "discussed",
}


def terms(text):
    """Distinctive terms from an expected answer, longest first."""
    words = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text.lower())
    found = {w for w in words if w not in STOPWORDS and len(w) > 3}
    return sorted(found, key=len, reverse=True)


def page_texts(pdf_path):
    reader = PdfReader(pdf_path)
    return [
        {"page": i + 1, "text": (p.extract_text() or "").lower()}
        for i, p in enumerate(reader.pages)
    ]


def score_pages(pages, wanted):
    """Score each page by how many of the wanted terms it contains."""
    scores = Counter()
    for page in pages:
        hits = sum(1 for term in wanted if term in page["text"])
        if hits:
            scores[page["page"]] = hits
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--file", default="eval_set.json")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--write", action="store_true",
                        help="write the single best page into expected_pages")
    args = parser.parse_args()

    pages = page_texts(args.pdf)
    print(f"Read {len(pages)} pages from {args.pdf}\n")

    with open(args.file, encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        if case["type"] != "answerable":
            continue

        wanted = terms(case.get("expected_answer", "") + " " + case["question"])
        scores = score_pages(pages, wanted)
        best = scores.most_common(args.top)

        print(f"Q: {case['question']}")
        print(f"   expected: {case.get('expected_answer', '')[:90]}")
        if case.get("section"):
            print(f"   section:  {case['section']}")

        if best:
            summary = ", ".join(f"p{page} ({hits} terms)" for page, hits in best)
            print(f"   likely:   {summary}")
            if args.write:
                case["expected_pages"] = [best[0][0]]
        else:
            print("   likely:   no match - check the wording of expected_answer")

        print()

    if args.write:
        with open(args.file, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        print(f"Wrote best guesses into {args.file}.")
        print("These are GUESSES. Open the PDF and verify each one before "
              "trusting any eval number produced from them.")


if __name__ == "__main__":
    main()
