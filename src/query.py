"""Ask questions about your ingested documents using a local LLM (Ollama).

Usage:
    python src/query.py                      # interactive mode
    python src/query.py "What is this about?"
    python src/query.py --model qwen2.5:3b "Summarize the document"
"""
import argparse

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION = "docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM = "llama3.2:3b"
TOP_K = 4  # how many chunks to retrieve

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the documents."

Context:
{context}

Question: {question}
Answer:"""


def retrieve(question: str, model: SentenceTransformer, collection, k: int = TOP_K):
    """Find the k chunks most similar to the question."""
    q_emb = model.encode([question]).tolist()
    result = collection.query(query_embeddings=q_emb, n_results=k)
    return result["documents"][0], result["metadatas"][0]


def ask(question: str, llm: str, embedder: SentenceTransformer, collection) -> None:
    docs, metas = retrieve(question, embedder, collection)
    context = "\n\n---\n\n".join(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    print()
    stream = ollama.chat(
        model=llm,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for part in stream:
        print(part["message"]["content"], end="", flush=True)

    sources = sorted({f"{m['source']} (chunk {m['chunk']})" for m in metas})
    print("\n\nSources:", ", ".join(sources), "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your documents locally.")
    parser.add_argument("question", nargs="?", help="Your question (omit for interactive mode)")
    parser.add_argument("--model", default=DEFAULT_LLM, help=f"Ollama model (default: {DEFAULT_LLM})")
    args = parser.parse_args()

    embedder = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION)

    if args.question:
        ask(args.question, args.model, embedder, collection)
        return

    print(f"Interactive mode with '{args.model}'. Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        ask(question, args.model, embedder, collection)


if __name__ == "__main__":
    main()
