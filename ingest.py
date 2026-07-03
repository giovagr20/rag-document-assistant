import os
from pathlib import Path


from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


DOCS_PATH = "docs"
DB_PATH = "chroma_db"


def load_documents():
    documents = []

    for file in os.listdir(DOCS_PATH):
        file_path = os.path.join(DOCS_PATH, file)

        if file.endswith(".pdf"):
            print(f"Cargando PDF: {file}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            print(f"Cargando TXT: {file}")
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

        elif file.endswith(".md"):
            print(f"Cargando Markdown: {file}")
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

        else:
            print(f"Saltando archivo no soportado: {file}")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)


def create_vector_db(chunks):
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    return db


def ingest():
    print("Cargando documentos...")
    documents = load_documents()

    if not documents:
        print("No se encontraron documentos.")
        return

    print(f"Documentos cargados: {len(documents)}")

    print("Dividiendo en chunks...")
    chunks = split_documents(documents)

    print(f"Chunks generados: {len(chunks)}")

    print("Creando embeddings...")
    create_vector_db(chunks)

    print("Ingesta completada correctamente.")


if __name__ == "__main__":
    ingest()