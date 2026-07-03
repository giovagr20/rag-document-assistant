from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

DB_PATH = "chroma_db"


def ask_question(query):
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    prompt = f"""
Answer using ONLY the context.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    while True:
        question = input("Pregunta: ")

        if question.lower() == "exit":
            break

        answer = ask_question(question)

        print("\nRespuesta:")
        print(answer)
        print()