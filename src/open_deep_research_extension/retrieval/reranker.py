from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def rerank_documents(query, documents, top_k=5):
    llm = ChatOpenAI(temperature=0)
    scored_docs = []
    for doc in documents:
        prompt = f"Rate this document for relevance to '{query}':\n\n{doc.page_content}\n\nScore (0-10):"
        try:
            score = int(llm.predict(prompt).strip())
        except:
            score = 0
        scored_docs.append((doc, score))
    return [doc for doc, _ in sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]]
