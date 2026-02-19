# src/utils/document_loader.py

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.docstore.document import Document
import os

def load_document(file_path: str) -> List[Document]:
    """
    Charge un document depuis un chemin de fichier en utilisant le chargeur approprié.
    Prend en charge les fichiers .pdf, .txt et .md.
    """
    file_extension = os.path.splitext(file_path)[1]
    
    if file_extension.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension.lower() in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Type de fichier non supporté: {file_extension}")
        
    return loader.load()

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Découpe une liste de documents en plus petits morceaux.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)
