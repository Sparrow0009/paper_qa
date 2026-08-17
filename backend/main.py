from fastapi import FastAPI, UploadFile
from pypdf import PdfReader
import io
from chunking import chunk_pages, strip_references
from store import store_document, list_documents, delete_document
from pydantic import BaseModel
from answer import ask
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile):
    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))
    pages = []
    for i,p in enumerate(reader.pages):
        pages.append({"page": i+1, "text": p.extract_text()})
    body, references_start = strip_references(pages)
    chunks = chunk_pages(body)
    doc_id, chunk_count = store_document(file.filename, chunks)
    return {
        "document_id": doc_id,
        "page_count": len(pages),
        "indexed_pages": len(body),
        "references_start": references_start,
        "chunk_count": len(chunks),
    }
         

class Question(BaseModel):
    question: str
    document_id: int | None = None

@app.post("/ask")
def ask_endpoint(q: Question):
    return ask(q.question, q.document_id)

@app.get("/documents")
def get_document():
    return list_documents()

@app.delete("/documents/{document_id}")
def delete_doc(document_id: int):
    if not delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found!")
    return {"deleted": document_id}
