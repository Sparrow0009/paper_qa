"""Exhaustive-question handling via map-reduce over every chunk.

Top-k retrieval cannot answer questions like "summarise this paper" or "list
everything covered", because it only ever sees k chunks. Semantically those
questions match the abstract, introduction and conclusion, so the model ends up
describing the front matter and presenting it as the whole document.

This module takes the other approach: read every chunk in batches (map), then
combine the batch findings into one answer (reduce). Slower and more expensive
than retrieval, so it only runs for questions that actually need it.
"""

import re
from concurrent.futures import ThreadPoolExecutor

from embeddings import client

MODEL = "gpt-4o-mini"

# The map-reduce path makes ~14 model calls per question. At the default
# temperature each one samples differently, so the same question produced
# noticeably different answers between runs and eval scores would be noise.
TEMPERATURE = 0

# Chunks per map call. ~20 chunks of 500 words is well inside the context
# window and keeps each call fast enough to parallelise.
BATCH_SIZE = 20
MAX_WORKERS = 6

# Guard against a huge corpus quietly costing a lot. 400 chunks of 500 words
# is roughly a 250 page document.
MAX_CHUNKS = 400

_EXHAUSTIVE_PATTERNS = [
    r"\ball\b.*\b(points?|topics?|sections?|techniques?|methods?|findings?)\b",
    r"\b(list|enumerate)\b.*\ball\b",
    r"\beverything\b",
    r"\bexhaustive\b",
    r"\bfull (list|summary|overview)\b",
    r"\bsummar(ise|ize|y)\b.*\b(document|paper|it|this)\b",
    r"\b(explain|describe|overview of|tell me about)\b.*\b(whole|entire|this document|this paper)\b",
    r"\bwhat is this (document|paper) about\b",
    r"\bmain (points?|themes?|arguments?|sections?)\b",
]


def is_exhaustive(question):
    """Heuristic routing.

    A cheap LLM classification call would generalise better, but it adds a
    round trip to every question and can itself be wrong. Regex is transparent,
    free and easy to explain - and a false negative just falls back to normal
    retrieval, which is a soft failure.
    """
    q = question.lower()
    return any(re.search(pattern, q) for pattern in _EXHAUSTIVE_PATTERNS)


def _batch(chunks, size):
    for i in range(0, len(chunks), size):
        yield chunks[i:i + size]


def _format(chunks):
    return "\n\n".join(
        f"[{c['filename']}, Page {c['page']}]\n{c['text']}" for c in chunks
    )


MAP_PROMPT = """You are reading one section of a longer document, in order to help
answer a specific question about it.

The question is:
{question}

Extract from these passages only the material that is relevant to that question.
For each point, give one or two sentences and the page it came from, in the form
[filename.pdf, Page 7].

Rules:
- If these passages contain nothing relevant to the question, reply with exactly:
  NOTHING RELEVANT
- Be thorough about what IS relevant. Include specific technique names, findings
  and limitations rather than generalising. Detail is the point.
- Only record what is actually in these passages. Do not infer or generalise.
- If a passage reports what another study, survey or reference did, say so
  explicitly, for example "Citing Smith et al., the paper notes that...".
  Do not present another work's findings as the document's own.
- The passages contain the document's bibliography markers such as [99] or
  [4, 6]. Ignore them completely and never reproduce them.
- If a passage is only references, acknowledgements, headers or boilerplate,
  return nothing for it.
- Plain text notes only, one point per line. These notes are an intermediate
  step, not the final answer, so do not format them."""

REDUCE_PROMPT = """You are given notes extracted from every section of a document,
in order, each with page citations. Only sections relevant to the question were
kept, so nearly all of this material should appear in your answer.

Answer the user's question using these notes. Because the notes were drawn from
the whole document, you can and should be comprehensive - do not compress the
detail away.

Rules:
- Merge duplicates and organise the material logically, following the document's
  own structure where it is apparent.
- Retain specific names, techniques and findings from the notes. Do not reduce
  several distinct techniques to one generic sentence.
- Keep the page citations, in the form [filename.pdf, Page 7]. Cite each page
  separately - never merge pages into a range such as "Page 12-13", and never
  invent a page.
- Preserve any attribution to other studies. Do not turn "citing Smith et al."
  into the document's own claim.
- Do not add anything that is not in the notes.
- Format using markdown. Short paragraphs, with numbered or bulleted lists where
  they genuinely aid clarity. Do not use headings."""


def _map_one(batch, question):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": MAP_PROMPT.format(question=question)},
            {"role": "user", "content": _format(batch)},
        ],
    )
    return response.choices[0].message.content


def _cited_pages(answer, chunks):
    """Pages the answer actually cites, so the sources panel reflects the answer
    rather than listing every page in the document.

    Handles "Page 7", "Pages 26-28" and "Pages 26, 27" - the model still emits
    ranges occasionally despite the prompt, and a cited page missing from the
    sources panel is exactly the kind of silent mismatch this project is
    supposed to rule out.
    """
    filenames = {c["filename"] for c in chunks}
    available = {(c["filename"], c["page"]) for c in chunks}
    cited = set()

    for filename in filenames:
        pattern = re.escape(filename) + r",?\s*Pages?\s*([\d\s,\-–—]+)"
        for group in re.findall(pattern, answer, flags=re.IGNORECASE):
            for part in re.split(r"[,\s]+", group.strip()):
                if not part:
                    continue
                span = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", part)
                if span:
                    start, end = int(span.group(1)), int(span.group(2))
                    if 0 < end - start < 50:
                        cited.update((filename, p) for p in range(start, end + 1))
                elif part.isdigit():
                    cited.add((filename, int(part)))

    return {c for c in cited if c in available}


def answer_exhaustive(question, chunks):
    """Map-reduce over the supplied chunks. Returns the same shape as ask()."""
    if not chunks:
        return {
            "answer": "There are no documents to summarise.",
            "sources": [],
            "mode": "exhaustive",
        }

    truncated = len(chunks) > MAX_CHUNKS
    working = chunks[:MAX_CHUNKS]
    batches = list(_batch(working, BATCH_SIZE))

    # Batches are independent, so run them concurrently - otherwise a 250 chunk
    # document is a dozen sequential API calls and the user waits a minute.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        notes = list(pool.map(lambda b: _map_one(b, question), batches))

    # Batches that found nothing relevant are dropped, so a narrow question
    # ("everything about hardware security") reduces over dense, on-topic notes
    # instead of 37 pages of mostly irrelevant material.
    kept = [n for n in notes if n and n.strip() and "NOTHING RELEVANT" not in n]
    combined = "\n\n".join(kept)

    if not combined:
        return {
            "answer": "I couldn't find anything relevant in the uploaded documents.",
            "sources": [],
            "mode": "exhaustive",
        }

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": REDUCE_PROMPT},
            {"role": "user", "content": f"Notes:\n{combined}\n\nQuestion:\n{question}"},
        ],
    )

    answer = response.choices[0].message.content
    if truncated:
        answer += (
            f"\n\nNote: this covers the first {MAX_CHUNKS} passages of a larger "
            "document and may not include the final sections."
        )

    # Show only the pages the answer actually cites. Listing all 37 pages of the
    # document implied every page was a source, which is exactly the kind of
    # unverifiable citation this project exists to avoid.
    cited = _cited_pages(answer, working)
    by_page = {}
    for chunk in working:
        by_page.setdefault((chunk["filename"], chunk["page"]), chunk["text"])

    sources = [
        {"filename": f, "page": p, "text": by_page.get((f, p), ""), "similarity": None}
        for f, p in sorted(cited)
    ]

    return {
        "answer": answer,
        "sources": sources,
        "mode": "exhaustive",
        "chunks_read": len(working),
        "batches_read": len(batches),
        "batches_relevant": len(kept),
    }
