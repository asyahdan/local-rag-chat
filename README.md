# local-rag-chat

Chat with your own documents (PDF, TXT, MD) using a local LLM. Everything runs on your machine, no API keys needed.

## How it works
1. `ingest.py` splits documents into chunks, embeds them, and stores them in ChromaDB.
2. `query.py` embeds your question, retrieves the most relevant chunks, and asks a local LLM (via Ollama) to answer from them.

## Setup
```bash
# 1. Install Ollama (https://ollama.com), then pull a model
ollama pull llama3.2:3b

# 2. Python environment
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
# Put your PDFs/TXT/MD files in data/, then:
python src/ingest.py
python src/query.py "What is this document about?"
python src/query.py            # interactive mode
python src/query.py --model qwen2.5:3b "Summarize the key points"
```

## Stack
Python, Ollama, ChromaDB, sentence-transformers (`all-MiniLM-L6-v2`)

## Roadmap
- [ ] Streamlit UI
- [ ] Evaluate answer quality with a test question set
- [ ] Support more file types
