from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

class ChromaVectorStore:
    def __init__(self, persist_directory="chroma_db"):
        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings()

    def create(self, documents):
        vectordb = Chroma.from_documents(documents, embedding=self.embedding, persist_directory=self.persist_directory)
        vectordb.persist()
        return vectordb

    def load(self):
        return Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)