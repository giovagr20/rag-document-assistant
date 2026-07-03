# RAG Guide

## What is RAG?

RAG stands for Retrieval Augmented Generation.

It improves LLM responses by retrieving relevant external knowledge before generating an answer.

## Main Components

### Documents
The source knowledge such as PDFs, markdown files, text files, or databases.

### Chunking
Large documents are split into smaller chunks.

### Embeddings
Embeddings convert text into vectors.

### Vector Database
Embeddings are stored in vector databases such as Chroma, Pinecone, or FAISS.

### Retrieval
When a user asks a question, the system retrieves the most relevant chunks.

### LLM
The language model uses the retrieved context to answer accurately.