"""Evaluation harness for the document Q&A system.

Measures two things separately:
  1. Retrieval recall@k - did the correct page appear in the top k chunks?
  2. Refusal rate      - does the system decline questions the documents cannot answer?

Retrieval is graded on its own because a wrong answer has two possible causes:
the retriever never found the passage, or it found it and the model misread it.
Grading only the final answer cannot tell those apart.
"""

import argparse
import json
from collections import defaultdict

from answer import ask
from embeddings import embed
from store import search

REFUSAL_MARKERS = [
    "I don't know based on the provided documents",
    "couldn't find anything relevant",
]


def is_refusal(answer_text):
    return any(marker.lower() in answer_text.lower() for marker in REFUSAL_MARKERS)


def evaluate_retrieval(cases, k):
    """Check whether an expected page appears in the top k retrieved chunks."""
    graded = [c for c in cases if c.get("expected_pages")]
    skipped = len(cases) - len(graded)

    hits = 0
    by_difficulty = defaultdict(lambda: [0, 0])
    by_style = defaultdict(lambda: [0, 0])
    misses = []

    for case in graded:
        results = search(embed([case["question"]])[0], limit=k)
        found_pages = {r["page"] for r in results}
        expected = set(case["expected_pages"])
        hit = bool(found_pages & expected)

        difficulty = case.get("difficulty", "unknown")
        # "source-worded" questions reuse the paper's own vocabulary, so keyword
        # overlap alone can carry them. "paraphrased" ones share no wording with
        # the passage, which is the honest test of semantic retrieval.
        style = case.get("style", "source-worded")
        by_difficulty[difficulty][1] += 1
        by_style[style][1] += 1
        if hit:
            hits += 1
            by_difficulty[difficulty][0] += 1
            by_style[style][0] += 1
        else:
            misses.append(
                {
                    "question": case["question"],
                    "expected": sorted(expected),
                    "retrieved": sorted(found_pages),
                    "top_score": results[0]["similarity"] if results else 0.0,
                }
            )

    return {
        "graded": len(graded),
        "skipped": skipped,
        "hits": hits,
        "by_difficulty": dict(by_difficulty),
        "by_style": dict(by_style),
        "misses": misses,
    }


def evaluate_refusals(cases):
    """Check that unanswerable questions are declined rather than answered."""
    correct = 0
    failures = []

    for case in cases:
        result = ask(case["question"])
        if is_refusal(result["answer"]) or not result["sources"]:
            correct += 1
        else:
            failures.append(
                {
                    "question": case["question"],
                    "answer": result["answer"][:160],
                    "top_score": result["sources"][0]["similarity"],
                }
            )

    return {"total": len(cases), "correct": correct, "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="eval_set.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--skip-refusals", action="store_true",
                        help="retrieval only - no LLM calls, so free and fast")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        cases = json.load(f)

    answerable = [c for c in cases if c["type"] == "answerable"]
    unanswerable = [c for c in cases if c["type"] == "unanswerable"]

    print(f"Loaded {len(cases)} cases "
          f"({len(answerable)} answerable, {len(unanswerable)} unanswerable)\n")

    retrieval = evaluate_retrieval(answerable, args.k)

    print("=" * 60)
    print(f"RETRIEVAL  (recall@{args.k})")
    print("=" * 60)

    if retrieval["skipped"]:
        print(f"Skipped {retrieval['skipped']} case(s) with no expected_pages filled in.\n")

    if retrieval["graded"]:
        pct = retrieval["hits"] / retrieval["graded"]
        print(f"Overall: {retrieval['hits']}/{retrieval['graded']} ({pct:.0%})\n")

        for difficulty in ("easy", "medium", "hard"):
            if difficulty in retrieval["by_difficulty"]:
                hit, total = retrieval["by_difficulty"][difficulty]
                print(f"  {difficulty:<14} {hit}/{total} ({hit / total:.0%})")

        if len(retrieval["by_style"]) > 1:
            print()
            for style in ("source-worded", "paraphrased"):
                if style in retrieval["by_style"]:
                    hit, total = retrieval["by_style"][style]
                    print(f"  {style:<14} {hit}/{total} ({hit / total:.0%})")

        if retrieval["misses"]:
            print(f"\n  Misses ({len(retrieval['misses'])}):")
            for miss in retrieval["misses"]:
                print(f"    - {miss['question']}")
                print(f"      expected {miss['expected']}, got {miss['retrieved']} "
                      f"(top score {miss['top_score']:.2f})")
    else:
        print("Nothing graded - fill in expected_pages first.")

    if args.skip_refusals:
        print("\nSkipping refusal checks (--skip-refusals).")
        return

    refusals = evaluate_refusals(unanswerable)

    print("\n" + "=" * 60)
    print("REFUSALS  (unanswerable questions)")
    print("=" * 60)
    pct = refusals["correct"] / refusals["total"] if refusals["total"] else 0
    print(f"Correctly declined: {refusals['correct']}/{refusals['total']} ({pct:.0%})")

    if refusals["failures"]:
        print(f"\n  Hallucinations ({len(refusals['failures'])}):")
        for failure in refusals["failures"]:
            print(f"    - {failure['question']}")
            print(f"      top score {failure['top_score']:.2f}: {failure['answer']}...")


if __name__ == "__main__":
    main()
