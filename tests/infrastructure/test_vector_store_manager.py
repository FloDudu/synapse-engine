# tests/infrastructure/test_vector_store_manager.py

import pytest
from unittest.mock import patch, MagicMock
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain.docstore.document import Document
from src.infrastructure.vector_store_manager import VectorStoreManager

# Mock settings avant toute importation qui pourrait les utiliser
@pytest.fixture(autouse=True)
def override_settings(monkeypatch):
    """Override settings to use a temporary directory for tests."""
    monkeypatch.setattr('src.config.settings.VECTOR_DB_PATH', './test_chroma_db')
    monkeypatch.setattr('src.config.settings.COLLECTION_NAME', 'test_collection')

@pytest.fixture
def fake_embedding_model():
    """Fixture pour un modèle d'embedding factice."""
    return FakeEmbeddings(size=768)

@patch('src.infrastructure.vector_store_manager.Chroma')
def test_singleton_behavior(MockChroma, fake_embedding_model):
    """Teste que le VectorStoreManager se comporte comme un singleton."""
    mock_chroma_instance = MagicMock()
    MockChroma.return_value = mock_chroma_instance

    # Forcer la réinitialisation du singleton pour ce test
    VectorStoreManager._instance = None
    
    manager1 = VectorStoreManager(embedding_model=fake_embedding_model)
    manager2 = VectorStoreManager(embedding_model=fake_embedding_model)

    assert manager1 is manager2
    # Chroma ne doit être appelé qu'une seule fois
    MockChroma.assert_called_once()

@patch('src.infrastructure.vector_store_manager.Chroma')
def test_add_documents(MockChroma, fake_embedding_model):
    """Teste l'ajout de documents."""
    mock_chroma_instance = MagicMock()
    MockChroma.return_value = mock_chroma_instance
    
    VectorStoreManager._instance = None
    manager = VectorStoreManager(embedding_model=fake_embedding_model)
    
    docs = [Document(page_content="Test document.")]
    manager.add_documents(docs)

    # Vérifie que la méthode add_documents de Chroma a été appelée
    mock_chroma_instance.add_documents.assert_called_once_with(docs)

@patch('src.infrastructure.vector_store_manager.Chroma')
def test_get_retriever(MockChroma, fake_embedding_model):
    """Teste la récupération du retriever."""
    mock_chroma_instance = MagicMock()
    MockChroma.return_value = mock_chroma_instance

    VectorStoreManager._instance = None
    manager = VectorStoreManager(embedding_model=fake_embedding_model)
    
    retriever = manager.get_retriever()

    # Vérifie que as_retriever a été appelé sur l'instance de Chroma
    mock_chroma_instance.as_retriever.assert_called_once()
    assert retriever is not None

def test_manager_raises_on_chroma_error():
    """Teste que le manager propage les exceptions de Chroma à l'init."""
    with patch('src.infrastructure.vector_store_manager.Chroma', side_effect=Exception("Chroma init failed")):
        VectorStoreManager._instance = None
        with pytest.raises(Exception, match="Chroma init failed"):
            VectorStoreManager(embedding_model=fake_embedding_model)

