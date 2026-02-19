# tests/utils/test_document_loader.py

import os
import pytest
from unittest.mock import patch, MagicMock
from langchain.docstore.document import Document
from src.utils.document_loader import load_document, chunk_documents

@pytest.fixture
def temp_files(tmp_path):
    """Crée des fichiers de test temporaires."""
    txt_content = "Ceci est un fichier texte simple." * 100
    md_content = "# Ceci est un titre\n\nCeci est un fichier markdown." * 50
    
    txt_file = tmp_path / "test.txt"
    txt_file.write_text(txt_content, encoding='utf-8')
    
    md_file = tmp_path / "test.md"
    md_file.write_text(md_content, encoding='utf-8')
    
    # Créer un faux fichier PDF vide juste pour le chemin
    pdf_file = tmp_path / "test.pdf"
    pdf_file.touch()

    return {"txt": str(txt_file), "md": str(md_file), "pdf": str(pdf_file)}

def test_load_document_txt(temp_files):
    """Teste le chargement d'un fichier .txt."""
    docs = load_document(temp_files["txt"])
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Ceci est un fichier texte simple." in docs[0].page_content

def test_load_document_md(temp_files):
    """Teste le chargement d'un fichier .md."""
    docs = load_document(temp_files["md"])
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "# Ceci est un titre" in docs[0].page_content

@patch("src.utils.document_loader.PyPDFLoader")
def test_load_document_pdf(mock_pypdfloader, temp_files):
    """Teste le chargement d'un fichier .pdf en mockant le loader."""
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = [Document(page_content="Contenu du PDF mocké.")]
    mock_pypdfloader.return_value = mock_loader_instance

    docs = load_document(temp_files["pdf"])
    
    mock_pypdfloader.assert_called_once_with(temp_files["pdf"])
    mock_loader_instance.load.assert_called_once()
    assert len(docs) == 1
    assert docs[0].page_content == "Contenu du PDF mocké."

def test_load_unsupported_file():
    """Teste le chargement d'un type de fichier non supporté."""
    with pytest.raises(ValueError, match="Type de fichier non supporté: .docx"):
        load_document("test.docx")

def test_chunk_documents():
    """Teste le découpage de documents."""
    docs = [
        Document(page_content="a" * 1500),
        Document(page_content="b" * 1500)
    ]
    
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=100)
    
    assert len(chunks) > 2  # Chaque document doit être découpé
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)
    # Vérifie le chevauchement
    assert chunks[0].page_content.endswith("a" * 100)
    assert chunks[1].page_content.startswith("a" * 100)

