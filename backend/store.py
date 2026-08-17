from db import get_connection
from embeddings import embed

def store_document(filename, chunks):
    vectors = embed([c["text"] for c in chunks])

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (filename) VALUES (%s) RETURNING id",
                (filename,),
            )
            doc_id = cur.fetchone()[0]

            for chunk, vector in zip(chunks, vectors):
                cur.execute(
                    "INSERT INTO chunks (document_id, page, text, embedding) VALUES  (%s, %s, %s, %s)",
                    (doc_id, chunk["page"], chunk["text"], vector),
                )
        conn.commit()

    return doc_id, len(chunks)

def search(question_vector, limit=10, document_id=None):
    sql="""
        SELECT d.filename, c.page, c.text, 1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
    """
    params = [question_vector]

    if document_id is not None:
        sql +=" WHERE c.document_id = %s"
        params.append(document_id)

    sql+=" ORDER BY c.embedding <=> %s::vector LIMIT %s"
    params.extend([question_vector, limit])

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql,params)
            rows = cur.fetchall()

    return [
        {"filename": r[0], "page": r[1], "text": r[2], "similarity": r[3]} 
        for r in rows
    ]


def list_documents():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, filename, uploaded_at FROM documents ORDER BY uploaded_at DESC"
            )
            rows = cur.fetchall()
    return [{"id": r[0], "filename": r[1], "uploaded_at": r[2]} for r in rows]


def delete_document(document_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            deleted = cur.rowcount
        conn.commit()
    return deleted


def get_all_chunks(document_id=None):
    sql = """
        SELECT d.filename, c.page, c.text
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
    """
    params = []

    if document_id is not None:
        sql += " WHERE c.document_id = %s"
        params.append(document_id)

    sql += " ORDER BY c.document_id, c.page, c.id"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [{"filename": r[0], "page": r[1], "text": r[2]} for r in rows]

