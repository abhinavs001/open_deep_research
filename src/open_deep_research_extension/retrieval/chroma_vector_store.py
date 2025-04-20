from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class ChromaVectorStore:
    def __init__(self, persist_directory="chroma_db"):
        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    def load(self):
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )
    