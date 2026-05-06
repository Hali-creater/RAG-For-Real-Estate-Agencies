import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document as LangchainDocument
from sqlalchemy.orm import Session
from models import Property, Document as DBDocument

class RAGEngine:
    def __init__(self, index_name="real_estate_index"):
        self.embeddings = OpenAIEmbeddings()
        self.index_name = index_name
        self.vector_store = None
        self.load_vector_store()

    def load_vector_store(self):
        if os.path.exists(self.index_name):
            self.vector_store = FAISS.load_local(self.index_name, self.embeddings, allow_dangerous_deserialization=True)
        else:
            # Initialize an empty vector store if possible, or wait until documents are added
            pass

    def save_vector_store(self):
        if self.vector_store:
            self.vector_store.save_local(self.index_name)

    def add_documents_from_db(self, db: Session):
        # Convert DB properties to Langchain documents
        properties = db.query(Property).all()
        docs = []
        for p in properties:
            content = f"Property: {p.title}\nDescription: {p.description}\nPrice: {p.price}\nLocation: {p.location}\nSize: {p.size}\nFeatures: {p.features}\nType: {p.property_type}"
            docs.append(LangchainDocument(page_content=content, metadata={"source": "db_property", "id": p.id}))

        # Convert DB documents (references to files) to Langchain documents
        db_docs = db.query(DBDocument).all()
        for d in db_docs:
            if os.path.exists(d.file_path):
                if d.file_path.endswith(".pdf"):
                    loader = PyPDFLoader(d.file_path)
                    docs.extend(loader.load())
                elif d.file_path.endswith(".txt"):
                    loader = TextLoader(d.file_path)
                    docs.extend(loader.load())

        if not docs:
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        if self.vector_store:
            self.vector_store.add_documents(split_docs)
        else:
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)

        self.save_vector_store()

    def query(self, query_text: str, k=4):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query_text, k=k)

rag_engine = RAGEngine()
