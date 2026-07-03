# RAG Assistant

AI assistant built with:

- LangChain
- OpenAI
- ChromaDB
- RAG Architecture

## Architecture

PDF Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Database (Chroma)
    ↓
Retriever
    ↓
LLM
    ↓
Answer

## Run

Install:

```bash
pip install -r requirements.txt
```

Index docs:

```bash
python ingest.py
```

Run assistant:

```bash
python app.py
```