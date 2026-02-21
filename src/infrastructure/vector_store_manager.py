# src/infrastructure/vector_store_manager.py

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain.docstore.document import Document
from typing import List
from src.config import settings
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Gère la création et l'accès à la base de données vectorielle.
    Utilise un design pattern Singleton pour garantir une seule instance.
    """
    _instance = None
    _vector_store = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, embedding_model: Embeddings):
        """
        Initialise la connexion à la base de données vectorielle, mais seulement si elle n'a pas déjà été faite.
        """
        # L'initialisation ne se fait qu'une seule fois.
        if self._vector_store is None:
            if embedding_model is None:
                logger.error("Le modèle d'embedding est requis pour la première initialisation.")
                return

            try:
                logger.info(f"Initialisation de la base de données vectorielle sur le disque à: {settings.VECTOR_DB_PATH}")
                self._vector_store = Chroma(
                    collection_name=settings.COLLECTION_NAME,
                    embedding_function=embedding_model,
                    persist_directory=settings.VECTOR_DB_PATH,
                )
                logger.info("Base de données vectorielle initialisée avec succès.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de ChromaDB: {e}")
                self._vector_store = None # S'assurer que l'état est cohérent en cas d'échec
                raise
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs=None):
        """
        Retourne un retriever pour la recherche de documents.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5} # Récupère les 5 chunks les plus pertinents par défaut
        
        if self._vector_store:
            return self._vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        else:
            logger.error("Le Vector Store n'est pas initialisé.")
            return None

    def add_documents(self, documents: List[Document]):
        """
        Ajoute des documents à la collection et s'assure de la persistance.
        """
        if self._vector_store is None:
            logger.error("Impossible d'ajouter des documents, le Vector Store n'est pas initialisé.")
            return

        if not documents:
            logger.warning("Aucun document à ajouter.")
            return
            
        logger.info(f"Ajout de {len(documents)} documents à la collection '{settings.COLLECTION_NAME}'...")
        try:
            self._vector_store.add_documents(documents)
            logger.info("Documents ajoutés avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout des documents à ChromaDB: {e}")
            raise
