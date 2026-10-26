"""Load documents from data/, split into chunks, embed them, store in ChromaDB.

Usage:
    python src/ingest.py
"""
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
DB_DIR = "chroma_db"
COLLECTION = "docs"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks


def read_file(path: Path) -> str:
    """Return the text of a PDF, .txt or .md file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks so context isn't cut mid-idea."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def main() -> None:
    files = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    if not files:
        print(f"No .pdf/.txt/.md files found in {DATA_DIR}/. Add some and run again.")
        return

    print(f"Loading embedding model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    for path in files:
        text = read_file(path)
        chunks = chunk_text(text)
        if not chunks:
            print(f"Skipping {path.name} (no extractable text)")
            continue

        embeddings = model.encode(chunks).tolist()
        collection.upsert(
            ids=[f"{path.name}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": path.name, "chunk": i} for i in range(len(chunks))],
        )
        print(f"Indexed {path.name}: {len(chunks)} chunks")

    print(f"Done. Collection now holds {collection.count()} chunks.")


if __name__ == "__main__":
    main()
