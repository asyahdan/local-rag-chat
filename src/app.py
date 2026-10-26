"""Streamlit UI for local-rag-chat.

Run from the project root:
    streamlit run src/app.py
"""
import chromadb
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

from ingest import DATA_DIR, DB_DIR, COLLECTION, EMBED_MODEL, chunk_text, read_file
from query import DEFAULT_LLM, PROMPT_TEMPLATE, retrieve

st.set_page_config(page_title="Local RAG Chat", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def list_models() -> list[str]:
    """Names of models installed in Ollama (falls back to the default)."""
    try:
        names = [m.model for m in ollama.list().models]
        return names or [DEFAULT_LLM]
    except Exception:
        return [DEFAULT_LLM]


def index_file(name: str, data: bytes) -> int:
    """Save an uploaded file to data/, chunk it, embed it, store it. Returns chunk count."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / name
    path.write_bytes(data)

    chunks = chunk_text(read_file(path))
    if not chunks:
        return 0

    embeddings = load_embedder().encode(chunks).tolist()
    load_collection().upsert(
        ids=[f"{name}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": name, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


def stream_answer(model: str, prompt: str):
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]


def show_sources(docs: list[str], metas: list[dict]) -> None:
    with st.expander("Sources"):
        for doc, meta in zip(docs, metas):
            st.markdown(f"**{meta['source']}** (chunk {meta['chunk']})")
            st.caption(doc[:400] + ("..." if len(doc) > 400 else ""))


# ---------- Sidebar ----------
embedder = load_embedder()
collection = load_collection()

with st.sidebar:
    st.header("Settings")
    models = list_models()
    default_index = models.index(DEFAULT_LLM) if DEFAULT_LLM in models else 0
    model = st.selectbox("LLM (Ollama)", models, index=default_index)
    top_k = st.slider("Chunks to retrieve", 1, 8, 4)

    st.divider()
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Index uploaded files", disabled=not uploads):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}..."):
                n = index_file(f.name, f.getvalue())
            if n:
                st.success(f"{f.name}: {n} chunks")
            else:
                st.warning(f"{f.name}: no extractable text")
    st.caption(f"Chunks in index: {collection.count()}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat ----------
st.title("💬 Chat with your documents")
st.caption("Runs fully on your machine with Ollama and ChromaDB.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("docs"):
            show_sources(msg["docs"], msg["metas"])

if question := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        count = collection.count()
        if count == 0:
            answer = "No documents indexed yet. Upload a file in the sidebar first."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            docs, metas = retrieve(question, embedder, collection, k=min(top_k, count))
            prompt = PROMPT_TEMPLATE.format(context="\n\n---\n\n".join(docs), question=question)
            try:
                answer = st.write_stream(stream_answer(model, prompt))
            except Exception as e:
                answer = f"Could not reach Ollama ({e}). Is the Ollama app running?"
                st.error(answer)
            else:
                show_sources(docs, metas)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "docs": docs, "metas": metas}
            )
