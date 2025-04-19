

from ingestion.pdf_docx_ingestor import LocalIngestor
from retrieval.chroma_vector_store import ChromaVectorStore

if __name__ == "__main__":
    ingestor = LocalIngestor()
    docs = ingestor.load_and_split("data/pdfs", "data/docx")
    ChromaVectorStore().create(docs)
    print("Database initialized with local documents.")

