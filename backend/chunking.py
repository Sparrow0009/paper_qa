import re

# A references section is a dense list of author names and paper titles. Indexed,
# it looks to a language model like a list of substantive points, and it produced
# false citations - claims sourced to a bibliography page, where the "finding"
# was really just the title of a cited paper. Cheaper and more reliable to drop
# it before it ever reaches the embeddings.
_REFERENCE_HEADING = re.compile(
    r"^\s*(?:\d+\s+)?(REFERENCES?|BIBLIOGRAPHY|WORKS CITED)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Only trust a heading found this far into the document, so a table of contents
# entry near the front cannot truncate the whole paper.
_MIN_POSITION = 0.5


def strip_references(pages, min_position=_MIN_POSITION):
    """Drop everything from the references heading onwards.

    Returns (pages, cut_at_page). cut_at_page is None if nothing was removed.
    """
    if not pages:
        return pages, None

    threshold = len(pages) * min_position

    for i, page in enumerate(pages):
        text = page.get("text") or ""
        if i >= threshold and _REFERENCE_HEADING.search(text):
            return pages[:i], page["page"]

    return pages, None


def chunk_pages(pages, chunk_size=500, overlap=50):
    """
    pages = [{"page": 1, "text": "..."}, ...]
    returns = [{"page": 1, "text": "<chunk>"},...]
    """
    chunks = []
    step = chunk_size - overlap
    for page in pages:
        text = page["text"]
        if not text:
            continue
        words = text.split()
        for start in range(0, len(words), step):
            piece = words[start:start + chunk_size]
            if not piece:
                continue
            if len(piece) < overlap and chunks:
                continue
            chunks.append({"page": page["page"], "text": " ".join(piece)})
    return chunks
