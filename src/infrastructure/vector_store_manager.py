# src/infrastructure/vector_store_manager.py

from abc import ABC, abstractmethod
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain.docstore.document import Document
from typing import List
from src.config import settings
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreStrategy(ABC):
    """Interface abstraite pour les stratégies de stockage vectoriel."""
    
    @abstractmethod
    def init_store(self, embedding_model: Embeddings):
        pass

    @abstractmethod
    def get_retriever(self, search_type: str, search_kwargs: dict):
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]):
        pass

class ChromaDBStrategy(VectorStoreStrategy):
    """Implémentation concrète pour ChromaDB."""
    
    def __init__(self):
        self._store = None

    def init_store(self, embedding_model: Embeddings):
        self._store = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=settings.VECTOR_DB_PATH,
        )

    def get_retriever(self, search_type: str, search_kwargs: dict):
        if not self._store:
            raise ValueError("Store non initialisé")
        return self._store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def add_documents(self, documents: List[Document]):
        if not self._store:
            raise ValueError("Store non initialisé")
        self._store.add_documents(documents)

class VectorStoreManager:
    """
    Gère la création et l'accès à la base de données vectorielle.
    Utilise un design pattern Singleton pour garantir une seule instance.
    Agit maintenant comme un Context pour la Strategy.
    """
    _instance = None
    _strategy: VectorStoreStrategy = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, embedding_model: Embeddings = None, strategy: VectorStoreStrategy = None):
        """
        Initialise le manager avec une stratégie spécifique (par défaut ChromaDB).
        """
        if self._initialized:
            return
            
        # Par défaut, on utilise la stratégie ChromaDB
        self._strategy = strategy if strategy else ChromaDBStrategy()
        
        if embedding_model:
            try:
                logger.info(f"Initialisation du Vector Store avec la stratégie : {self._strategy.__class__.__name__}")
                self._strategy.init_store(embedding_model)
                self._initialized = True
                logger.info("Base de données vectorielle initialisée avec succès.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du Vector Store: {e}")
                self._initialized = False
                raise
        else:
            logger.warning("VectorStoreManager instancié sans modèle d'embedding. En attente d'initialisation.")

    def get_retriever(self, search_type: str = "similarity", search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        if self._initialized and self._strategy:
            return self._strategy.get_retriever(search_type, search_kwargs)
        else:
            logger.error("Le Vector Store n'est pas initialisé.")
            return None

    def add_documents(self, documents: List[Document]):
        if not self._initialized or not self._strategy:
            logger.error("Impossible d'ajouter des documents, le Vector Store n'est pas initialisé.")
            return

        if not documents:
            logger.warning("Aucun document à ajouter.")
            return
            
        logger.info(f"Ajout de {len(documents)} documents via {self._strategy.__class__.__name__}...")
        try:
            self._strategy.add_documents(documents)
            logger.info("Documents ajoutés avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout des documents: {e}")
            raise
