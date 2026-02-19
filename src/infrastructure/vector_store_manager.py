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

    Utilise le design pattern 'Singleton' implicite pour la connexion à la DB.

    """

    _instance = None



    def __new__(cls, embedding_model: Embeddings):
        # On s'assure qu'une seule instance est créée.
        if cls._instance is None:
            instance = super(VectorStoreManager, cls).__new__(cls)
            try:
                logger.info(f"Initialisation de la base de données vectorielle sur le disque à: {settings.VECTOR_DB_PATH}")
                instance._vector_store = Chroma(
                    collection_name=settings.COLLECTION_NAME,
                    embedding_function=embedding_model,
                    persist_directory=settings.VECTOR_DB_PATH,
                )
                logger.info("Base de données vectorielle initialisée avec succès.")
                cls._instance = instance
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de ChromaDB: {e}")
                raise
        return cls._instance

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
        if not hasattr(self, '_vector_store') or not self._vector_store:
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
