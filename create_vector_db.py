from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.chroma import Chroma
from typing import List, Dict
from langchain.schema import Document
import shutil
import os

DATA_PATH = "docs"
CHROMA_PATH = "chroma"

def main():
    pass


def load_documents(path: str) -> List[Document]:
    loader = DirectoryLoader(path, glob="*.md")
    doc = loader.load()
    return doc
    
def split_text(document: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=75,
        length_function=len,
        add_start_index=True
        )
    chunks = text_splitter.split_documents(document)
    print("Splitting", len(document), " documents into ", len(chunks), " chunks.")
    return chunks


def add_to_chroma(documents: List[Document]):
    print("Check Chroma...")
    if os.path.exists("CHROMA_PATH"):
        shutil.rmtree("CHROMA_PATH")
    
    # embeddings = OllamaEmbeddings()
    print("Initiating the db...")
    db = Chroma.from_documents(
        documents,
        OllamaEmbeddings(model="llama3"),
        persist_directory=CHROMA_PATH
        )
    print("Starting persist...")
    db.persist()
    
    print(f"Saved {len(documents)} chunks to {CHROMA_PATH}")
    
  
def initiate_data_store():
    documents = load_documents(DATA_PATH)
    chunks = split_text(documents)
    add_to_chroma(chunks)
    


if __name__=="__main__":
    initiate_data_store()
    




