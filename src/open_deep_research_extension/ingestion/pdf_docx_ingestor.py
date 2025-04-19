from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class LocalIngestor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_pdfs(self, directory):
        docs = []
        for file in os.listdir(directory):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(directory, file))
                docs.extend(loader.load())
        return docs

    def load_docx(self, directory):
        docs = []
        for file in os.listdir(directory):
            if file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(os.path.join(directory, file))
                docs.extend(loader.load())
        return docs

    def load_and_split(self, pdf_dir, docx_dir):
        documents = self.load_pdfs(pdf_dir) + self.load_docx(docx_dir)
        return self.splitter.split_documents(documents)
