import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState("all");
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  async function loadDocuments() {
    try {
      const res = await fetch(`${API}/documents`);
      if (!res.ok) throw new Error(`Could not load documents (${res.status})`);
      setDocuments(await res.json());
    } catch (err) {
      setError(err.message);
    }
  }
  async function handleDelete() {
    if (selectedDoc === "all") return;
    const doc = documents.find((d) => String(d.id) === selectedDoc);
    if (!confirm(`Delete "${doc?.filename}" and all its chunks?`)) return;

    setError(null);
    try {
      const res = await fetch(`${API}/documents/${selectedDoc}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`Delete failed (${res.status})`);
      setSelectedDoc("all");
      await loadDocuments();
    } catch (err) {
      setError(err.message);
    }
  }
  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setError(null);
    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // No Content-Type header here on purpose - the browser sets the
      // multipart boundary itself, and overriding it breaks the upload.
      const res = await fetch(`${API}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Upload failed (${res.status})`);

      const data = await res.json();
      await loadDocuments();
      setSelectedDoc(String(data.document_id));
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          text: `Indexed ${file.name} - ${data.indexed_pages ?? data.page_count} of ${data.page_count} pages, ${data.chunk_count} chunks${
            data.references_start ? ` (references from p${data.references_start} skipped)` : ""
          }.`,
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleAsk(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || asking) return;

    setError(null);
    setMessages((prev) => [...prev, { role: "user", text: trimmed }]);
    setQuestion("");
    setAsking(true);

    try {
      const body = { question: trimmed };
      if (selectedDoc !== "all") body.document_id = Number(selectedDoc);

      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) throw new Error(`Request failed (${res.status})`);

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer, sources: data.sources || [] },
      ]);
    } catch (err) {
      setError(err.message);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Something went wrong answering that.", sources: [] },
      ]);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Paper QA</h1>
        <p className="subtitle">
          Ask questions across your documents. Every answer cites the passage it came from.
        </p>
      </header>

      <div className="controls">
        <label className={`upload-btn ${uploading ? "disabled" : ""}`}>
          {uploading ? "Indexing..." : "Upload PDF"}
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={handleUpload}
            disabled={uploading}
            hidden
          />
        </label>

        <select
          className="doc-select"
          value={selectedDoc}
          onChange={(e) => setSelectedDoc(e.target.value)}
        >
          <option value="all">All documents ({documents.length})</option>
          {documents.map((doc) => (
            <option key={doc.id} value={doc.id}>
              {doc.filename}
            </option>
          ))}
        </select>
        {selectedDoc !== "all" && (
          <button className="delete-btn" onClick={handleDelete}>
            Delete
          </button>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      <main className="chat">
        {messages.length === 0 && !asking && (
          <div className="empty">
            {documents.length === 0
              ? "Upload a PDF to get started."
              : "Ask a question about your documents."}
          </div>
        )}

        {messages.map((message, i) => (
          <Message key={i} message={message} />
        ))}

        {asking && <div className="thinking">Searching documents...</div>}
        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={handleAsk}>
        <input
          type="text"
          value={question}
          placeholder="Ask a question..."
          onChange={(e) => setQuestion(e.target.value)}
          disabled={asking}
        />
        <button type="submit" disabled={asking || !question.trim()}>
          Ask
        </button>
      </form>
    </div>
  );
}

function Message({ message }) {
  if (message.role === "system") {
    return <div className="system-msg">{message.text}</div>;
  }

  if (message.role === "user") {
    return <div className="msg user">{message.text}</div>;
  }

  return (
    <div className="msg assistant">
      <div className="answer">
        <ReactMarkdown>{message.text}</ReactMarkdown>
      </div>
      {message.sources?.length > 0 && (
        <div className="sources">
          <div className="sources-label">
            Sources ({message.sources.length})
          </div>
          {message.sources.map((source, i) => (
            <Source key={i} source={source} />
          ))}
        </div>
      )}
    </div>
  );
}

function Source({ source }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="source">
      <button className="source-head" onClick={() => setOpen(!open)}>
        <span className="chevron">{open ? "-" : "+"}</span>
        <span className="source-name">
          {source.filename} &middot; page {source.page}
        </span>
        <span className="score">
          {source.similarity == null
            ? "cited"
            : `${(source.similarity * 100).toFixed(0)}% match`}
        </span>
      </button>
      {open && <div className="source-text">{source.text}</div>}
    </div>
  );
} 