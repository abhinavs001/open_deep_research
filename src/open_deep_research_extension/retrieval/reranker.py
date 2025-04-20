from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from typing import List

def rerank_documents(query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
    """
    Re-rank documents using a language model to score their relevance to a given query.

    Args:
        query (str): The input question or topic for which relevance is being measured.
        documents (List[Document]): A list of langchain Document objects retrieved from a vector store.
        top_k (int): The number of top-ranked documents to return.

    Returns:
        List[Document]: A list of the top_k most relevant documents.
    """
    llm = ChatOpenAI(model="o3-mini", temperature=0)
    scored_docs = []
    for doc in documents:
        prompt = (
            f"Rate the relevance of the following document to the query:\n\n"
            f"Query: {query}\n\n"
            f"Document:\n{doc.page_content}\n\n"
            f"Respond with a single integer score from 0 (not relevant) to 10 (very relevant):"
        )
        try:
            score_text = llm.predict(prompt).strip()
            score = int("".join(filter(str.isdigit, score_text)))
        except Exception:
            score = 0
        scored_docs.append((doc, score))

    # Sort the documents by score in descending order and return the top_k
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:top_k]]
