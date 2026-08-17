from embeddings import client, embed
from overview import answer_exhaustive, is_exhaustive
from store import get_all_chunks, search

SYSTEM_PROMPT = """You answer questions about documents the user has uploaded.

Each passage in the context is preceded by a label of the form
[Filename example.pdf, Page 7]. That label is the only valid source of a citation.

Rules:
- Base your answer on the context provided. Do not add facts from outside knowledge.
- You may summarise, explain, synthesise and draw connections across the context. Broad requests like "explain this document" should be answered by summarising what the context contains.
- Only respond with "I don't know based on the provided documents." when the context is genuinely unrelated to the question. Do not refuse simply because the context is partial - answer with what is there.
- Every claim in your answer must end with a citation copied from the label above the passage you used, in the form [Filename example.pdf, Page 7]. If a claim draws on several passages, cite each of them.
- The passages contain the document's own bibliography markers, such as [99], [33] or [4, 6]. These are NOT citations and mean nothing to the reader. Never reproduce them in your answer. Cite only using the [Filename, Page N] labels.
- Never invent or infer a page range such as "Pages 1-30". Cite only pages that actually appear in the labels.
- Academic papers describe other people's work as well as their own. If a passage reports what another study, survey or reference did, attribute it to that other work, not to the authors of the document. Do not present findings from a related-work discussion as the document's own contribution.
- Format using markdown. Short paragraphs, with numbered or bulleted lists where they genuinely aid clarity. Do not use headings.
- When the context is clearly a subset of a larger document, note that your answer is based on the retrieved passages and may not cover everything."""

MIN_SIMILARITY = 0.3


def ask(question, document_id=None):
    # Exhaustive questions ("summarise this paper", "list everything covered")
    # cannot be answered by top-k retrieval, so they take the map-reduce path
    # over every chunk instead. See overview.py.
    if is_exhaustive(question):
        return answer_exhaustive(question, get_all_chunks(document_id))

    question_vector = embed([question])[0]
    results = search(question_vector, document_id=document_id)

    # Filter the whole list, not just the top hit, otherwise weak chunks at
    # positions 4 and 5 still get fed to the model as context.
    results = [r for r in results if r["similarity"] >= MIN_SIMILARITY]

    if not results:
        return {
            "answer": "I couldn't find anything relevant in the uploaded documents.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[Filename {r['filename']}, Page {r['page']}]\n{r['text']}" for r in results
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        # Grounded extraction, not creative writing; we want the most likely
        # output, and reproducible runs so eval scores mean something.
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": results,
    }