import pytest
from unittest.mock import patch, MagicMock
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain.docstore.document import Document
from src.infrastructure.vector_store_manager import VectorStoreManager


@pytest.fixture(autouse=True)
def override_settings(monkeypatch):
    monkeypatch.setattr('src.config.settings.VECTOR_DB_PATH', './test_chroma_db')
    monkeypatch.setattr('src.config.settings.COLLECTION_NAME', 'test_collection')


@pytest.fixture(autouse=True)
def reset_singleton():
    VectorStoreManager._instance = None
    yield
    VectorStoreManager._instance = None


@pytest.fixture
def fake_embedding_model():
    return FakeEmbeddings(size=768)


@patch('src.infrastructure.vector_store_manager.Chroma')
def test_singleton_behavior(MockChroma, fake_embedding_model):
    mock_chroma_instance = MagicMock()
    MockChroma.return_value = mock_chroma_instance

    manager1 = VectorStoreManager(embedding_model=fake_embedding_model)
    manager2 = VectorStoreManager(embedding_model=fake_embedding_model)

    assert manager1 is manager2
    MockChroma.assert_called_once()


@patch('src.infrastructure.vector_store_manager.Chroma')
def test_add_documents(MockChroma, fake_embedding_model):
    mock_chroma_instance = MagicMock()
    MockChroma.return_value = mock_chroma_instance

    manager = VectorStoreManager(embedding_model=fake_embedding_model)

    docs = [Document(page_content="Test document.")]
    manager.add_documents(docs)

    mock_chroma_instance.add_documents.assert_called_once_with(docs)


@patch('src.infrastructure.vector_store_manager.Chroma')
def test_get_retriever(MockChroma, fake_embedding_model):
    mock_chroma_instance = MagicMock()
    MockChroma.return_value = mock_chroma_instance

    manager = VectorStoreManager(embedding_model=fake_embedding_model)
    retriever = manager.get_retriever()

    mock_chroma_instance.as_retriever.assert_called_once()
    assert retriever is not None


def test_manager_raises_on_chroma_error(fake_embedding_model):
    with patch('src.infrastructure.vector_store_manager.Chroma', side_effect=Exception("Chroma init failed")):
        with pytest.raises(Exception, match="Chroma init failed"):
            VectorStoreManager(embedding_model=fake_embedding_model)
