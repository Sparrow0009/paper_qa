# Code Guide

A plain-English walkthrough of every file and function in this project.

---

## The big picture

The app answers questions about PDFs you upload, and shows which page each answer came from.

It works in two stages.

**When you upload a PDF (happens once):**

1. Pull the text out of the PDF
2. Cut the bibliography off the end
3. Chop the text into small chunks
4. Turn each chunk into a list of numbers that represents its meaning (an "embedding")
5. Save the chunks and their numbers in the database

**When you ask a question:**

1. Decide what kind of question it is
2. Normal question: find the handful of chunks closest in meaning, answer from those
3. "Summarise everything" question: read every chunk in batches, then combine

The reason for chunks: you can't paste a 30-page paper into the model for every question. It's too big, too slow and too expensive, and the answer gets vague. So we store the paper in pieces and fetch only the pieces we need.

---

## `docker-compose.yml` (project root)

Starts the database in a container, so you don't install Postgres on your machine. The image is `pgvector/pgvector`, which is Postgres with an add-on for storing and comparing embeddings.

## `backend/schema.sql`

Creates the two tables.

- **`documents`**: one row per uploaded PDF (id, filename, upload time)
- **`chunks`**: one row per chunk (which document it belongs to, page number, the text, the embedding)

`ON DELETE CASCADE` means deleting a document automatically deletes its chunks, so you never get orphaned rows.

---

## `backend/db.py`

**`get_connection()`**
Opens a connection to the database and registers the `vector` type so Python lists can be stored in embedding columns. Reads the address from an environment variable, falling back to the local Docker one, so deploying needs a config change rather than a code change.

---

## `backend/embeddings.py`

**`embed(texts, batch_size=100)`**
Sends a list of strings to OpenAI and gets back a list of numbers for each one (1536 numbers per text). Similar meanings produce similar numbers, which is what makes search work.
It sends them in batches of 100 rather than one at a time, because one request for 100 chunks is far faster and cheaper than 100 requests.

---

## `backend/chunking.py`

**`strip_references(pages)`**
Finds the REFERENCES heading and throws away everything after it.
Why: a bibliography is a list of paper titles, and the model mistook those titles for the paper's own findings, producing real-looking citations to made-up claims. It only trusts the heading if it appears in the back half of the document, so a table of contents at the front can't accidentally delete the whole paper.

**`chunk_pages(pages, chunk_size=500, overlap=50)`**
Splits each page into chunks of about 500 words, keeping the page number attached so citations can point back.
Each chunk repeats the last 50 words of the previous one (the "overlap") so a sentence split across a boundary still makes sense in both halves. Very short leftover chunks are dropped, because they're almost entirely overlap and add nothing.

---

## `backend/store.py`

Everything that touches the database lives here, so no other file writes SQL.

**`store_document(filename, chunks)`**
Saves a new document and all its chunks. Embeds every chunk in one go, inserts the document row, gets its new id back, then inserts each chunk against that id.

**`search(question_vector, limit=10, document_id=None)`**
The heart of retrieval. Asks the database for the chunks whose embeddings are closest to the question's embedding.
`<=>` is the "how different are these two lists of numbers" operator. Smaller means more similar, which is why results are sorted ascending. Pass a `document_id` to search one document, leave it out to search everything.

**`list_documents()`**
Returns every uploaded document, newest first. Used to fill the dropdown in the UI.

**`delete_document(document_id)`**
Deletes a document and returns how many rows were removed, so the API can tell the difference between "deleted" and "that id doesn't exist". Its chunks go automatically via the cascade rule.

**`get_all_chunks(document_id=None)`**
Returns every chunk in page order, with no searching or ranking involved. Used by the summarise path, which needs the whole document rather than the best matches.

---

## `backend/answer.py`

The normal question path, plus the decision about which path to use.

**`SYSTEM_PROMPT`**
The instructions sent to the model with every question. Tells it to answer only from the passages given, to cite the filename and page, not to copy the paper's own reference numbers like `[99]`, and to attribute other people's work correctly rather than presenting it as this paper's finding.

**`MIN_SIMILARITY`**
The cut-off score. Chunks below it are thrown away as too weakly related to be useful.

**`ask(question, document_id=None)`**
The main entry point.
1. If the question is asking for everything, hand it to the summarise path and stop.
2. Otherwise embed the question, find the closest chunks, drop the weak ones.
3. If nothing survives, say so instead of guessing.
4. Otherwise build the context with a `[filename, page]` label above each passage, send it with the question, return the answer plus the sources.

`temperature=0` tells the model to pick the most likely wording rather than varying it, so the same question gives the same answer twice.

---

## `backend/overview.py`

The summarise path, for questions like "list everything in this paper".

Why it exists: normal search finds the chunks closest in meaning to the question. But "summarise this paper" isn't about any one topic, so the closest chunks are the abstract and conclusion, and the model ends up describing the front matter while never seeing the middle of the paper. This path reads everything instead.

**`is_exhaustive(question)`**
Looks for phrases like "list all", "everything", "summarise this document", "main points". If found, use the summarise path.
It's simple word-matching rather than asking the model to decide, because it's instant and free, and a wrong guess just falls back to normal search, which still gives a decent answer.

**Constants at the top**
`BATCH_SIZE = 20` chunks per map call, `MAX_WORKERS = 6` batches running at once, `MAX_CHUNKS = 400` as a safety cap so a huge document can't quietly cost a lot, and `TEMPERATURE = 0` so repeated runs give the same answer.

**`_batch(chunks, size)`**
Splits a long list of chunks into groups of 20. The leading underscore is a convention meaning "internal helper, not meant to be used from other files".

**`_format(chunks)`**
Turns a group of chunks into one block of text, with a `[filename, page]` label above each passage so the model knows where each one came from.

**`MAP_PROMPT`**
Instructions for reading one batch. Asks for only the material relevant to the question, and to reply `NOTHING RELEVANT` if there's none. That's how irrelevant batches get filtered out, so a narrow question doesn't drown in 30 unrelated pages.

**`REDUCE_PROMPT`**
Instructions for the final step. Combine all the notes into one answer, merge duplicates, keep the citations, don't compress the detail away.

**`_map_one(batch, question)`**
Sends a single batch to the model and returns its notes.

**`_cited_pages(answer, chunks)`**
Reads the finished answer and pulls out every page it cited, so the sources panel shows those pages and only those.
Handles `Page 7`, `Pages 26-28` and `Pages 13, 14`, expanding ranges into individual pages. Then the important bit: it throws away any page that isn't actually in the database, so if the model invents a page number it never reaches the screen.

**`answer_exhaustive(question, chunks)`**
Runs the whole thing.
1. Split all the chunks into batches
2. Send every batch to the model at the same time (`ThreadPoolExecutor`) rather than one after another, which takes 6 seconds instead of a minute
3. Drop the batches that found nothing relevant
4. Send the surviving notes for the final combine
5. Work out which pages the answer cites and return those as sources

---

## `backend/main.py`

The web API. Each function below is a URL the frontend can call.

**`health()`** on `GET /health`
Returns `{"status": "ok"}`. A quick way to check the server is alive.

**`upload(file)`** on `POST /upload`
Takes a PDF, pulls the text out page by page, strips the references, chunks it, embeds it, saves it.
Returns the new document id, the total page count, `indexed_pages` (how many were actually chunked after the bibliography was removed), `references_start` (the page the cut happened on, or null), and the chunk count. The UI shows those in its confirmation line.

**`Question`**
Describes what an `/ask` request must contain: a question, and optionally a document id. FastAPI uses it to reject malformed requests automatically.

**`ask_endpoint(q)`** on `POST /ask`
Passes the question to `ask()` and returns the answer with its sources.

**`get_document()`** on `GET /documents`
Returns the list of uploaded documents for the dropdown.

**`delete_doc(document_id)`** on `DELETE /documents/{id}`
Deletes one document, or returns a 404 if that id doesn't exist.

**CORS middleware**
Browsers block a page on one address from calling a server on another. The frontend runs on port 5173 and the API on 8000, so this explicitly allows it.

---

## `backend/eval.py`

Measures how well the system works, so "does it work?" has a number instead of a shrug.

**`is_refusal(answer_text)`**
Checks whether an answer was a refusal ("I don't know based on the provided documents").

**`evaluate_retrieval(cases, k)`**
The main measurement. For each test question, searches and checks whether the page you know holds the answer appeared in the top k results.
It grades **retrieval on its own**, deliberately. If an answer is wrong, you need to know whether the search failed to find the passage or found it and the model misread it. Grading the final answer alone can't tell you which.

**`evaluate_refusals(cases)`**
Asks questions the documents can't answer and checks the system declines instead of inventing something.

**`main()`**
Loads the test set, runs both checks, prints the scores broken down by difficulty and phrasing style, and lists every miss with what it expected versus what it got, so failures can be diagnosed rather than just counted.

---

## `backend/fill_pages.py`

A helper for building the answer key in `eval_set.json`.

**`terms(text)`**
Pulls the distinctive words out of an expected answer, ignoring common ones like "the" and "paper".

**`page_texts(pdf_path)`**
Reads the PDF and returns the text of each page.

**`score_pages(pages, wanted)`**
Scores each page by how many of those distinctive words it contains. A high score means the answer is probably on that page.

**`main()`**
Prints the most likely pages for each test question, so you can check them in the PDF and fill in the answer key.

It reads the **PDF**, never the app. Building the answer key from the system's own output would be marking its own homework, scoring near 100% while measuring nothing.

---

## `backend/eval_set.json`

The test questions. Each has the question, the pages where the answer really is, a difficulty rating, and the correct answer written out.

Questions marked `unanswerable` are things the paper doesn't cover. They test that the system says "I don't know" instead of making something up.

Questions marked `"style": "paraphrased"` are written in everyday language with no wording in common with the source passage, which is the honest test of whether the embeddings are doing the work.

---

## `frontend/src/App.jsx`

The user interface: an upload button, a document dropdown, a chat box, and the sources under each answer.

**`App()`**
Holds the state (documents, messages, loading flags) and the functions that talk to the API: `loadDocuments`, `handleUpload`, `handleAsk` and `handleDelete`.

**`Message({ message })`**
Draws one message. Your questions render on the right, answers in a card with sources underneath.

**`Source({ source })`**
One source line: filename, page, match score. Click to expand the actual passage the answer came from.
This is the most important part of the interface. Without it, this is a chatbot. With it, every claim can be checked against the document.

---

## Reading order

If you're coming back to this cold, follow one question through the system:

1. `backend/main.py`, where the request arrives
2. `backend/answer.py`, which decides the path
3. `backend/store.py`, how chunks are found
4. `backend/overview.py`, what happens for summarise-everything questions
