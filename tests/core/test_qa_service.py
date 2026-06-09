import pytest
from unittest.mock import MagicMock, patch, call
from src.core.qa_service import QAService


@pytest.fixture
def mocked_deps():
    """Patch AIFactory and VectorStoreManager for the duration of a test."""
    with patch("src.core.qa_service.AIFactory") as MockFactory, \
         patch("src.core.qa_service.VectorStoreManager") as MockVSM:

        mock_llm = MagicMock()
        mock_embedding = MagicMock()
        mock_retriever = MagicMock()

        MockFactory.get_llm.return_value = mock_llm
        MockFactory.get_embedding_model.return_value = mock_embedding
        MockVSM.return_value.get_retriever.return_value = mock_retriever

        yield MockFactory, MockVSM, mock_llm, mock_retriever


def test_init_wires_components_correctly(mocked_deps):
    MockFactory, MockVSM, mock_llm, mock_retriever = mocked_deps

    service = QAService()

    MockFactory.get_llm.assert_called_once()
    MockFactory.get_embedding_model.assert_called_once()
    assert service.llm is mock_llm
    assert service.retriever is mock_retriever


def test_init_raises_runtime_error_on_missing_api_key(mocked_deps):
    MockFactory, *_ = mocked_deps
    MockFactory.get_llm.side_effect = ValueError("MISTRAL_API_KEY n'est pas définie.")

    with pytest.raises(RuntimeError, match="Erreur d'initialisation du service"):
        QAService()


def test_ask_question_invokes_chain_with_question(mocked_deps):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Test answer."

    service = QAService()
    with patch.object(service, "get_rag_chain", return_value=mock_chain):
        answer = service.ask_question("What is RAG?")

    mock_chain.invoke.assert_called_once_with("What is RAG?")
    assert answer == "Test answer."


def test_rag_chain_is_built_only_once_across_calls(mocked_deps):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "answer"

    service = QAService()
    with patch.object(service, "get_rag_chain", return_value=mock_chain) as mock_build:
        service.ask_question("Q1")
        service.ask_question("Q2")

    mock_build.assert_called_once()
    assert mock_chain.invoke.call_count == 2
