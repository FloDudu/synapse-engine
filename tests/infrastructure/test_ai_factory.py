import pytest
from unittest.mock import patch
from langchain_mistralai import ChatMistralAI
from langchain_voyageai import VoyageAIEmbeddings

from src.infrastructure.ai_factory import AIFactory


def test_get_embedding_model_raises_error_if_key_is_missing(mocker):
    """
    Vérifie que ValueError est levé si VOYAGE_API_KEY n'est pas configurée.
    """
    mocker.patch('src.config.settings.VOYAGE_API_KEY', None)
    with pytest.raises(ValueError, match="VOYAGE_API_KEY n'est pas définie."):
        AIFactory.get_embedding_model()


def test_get_llm_raises_error_if_key_is_missing(mocker):
    """
    Vérifie que ValueError est levé si MISTRAL_API_KEY n'est pas configurée.
    """
    mocker.patch('src.config.settings.MISTRAL_API_KEY', None)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY n'est pas définie."):
        AIFactory.get_llm()


def test_get_embedding_model_returns_correct_instance(mocker):
    """
    Vérifie que la factory retourne une instance de VoyageAIEmbeddings
    quand la clé API est présente.
    """
    mocker.patch('src.config.settings.VOYAGE_API_KEY', 'fake-api-key')
    embedding_model = AIFactory.get_embedding_model()
    assert isinstance(embedding_model, VoyageAIEmbeddings)


def test_get_llm_returns_correct_instance(mocker):
    """
    Vérifie que la factory retourne une instance de ChatMistralAI
    quand la clé API est présente.
    """
    mocker.patch('src.config.settings.MISTRAL_API_KEY', 'fake-api-key')
    llm = AIFactory.get_llm()
    assert isinstance(llm, ChatMistralAI)
