# src/ingest.py

import os
import argparse
import logging
from typing import List
from langchain.docstore.document import Document

from src.utils.document_loader import load_document, chunk_documents
from src.infrastructure.ai_factory import AIFactory
from src.infrastructure.vector_store_manager import VectorStoreManager

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_directory(directory_path: str):
    """
    Ingère tous les documents d'un répertoire dans la base de données vectorielle.
    """
    if not os.path.isdir(directory_path):
        logger.error(f"Le chemin spécifié n'est pas un répertoire valide: {directory_path}")
        return

    supported_extensions = [".pdf", ".txt", ".md"]
    all_chunks: List[Document] = []

    logger.info(f"Début de l'ingestion du répertoire: {directory_path}")

    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1]

            if file_extension.lower() in supported_extensions:
                try:
                    logger.info(f"Chargement du document: {file_path}")
                    docs = load_document(file_path)
                    
                    logger.info(f"Découpage de {len(docs)} document(s) en chunks...")
                    chunks = chunk_documents(docs)
                    all_chunks.extend(chunks)
                    logger.info(f"{len(chunks)} chunks créés.")

                except Exception as e:
                    logger.error(f"Erreur lors du traitement du fichier {file_path}: {e}")
    
    if not all_chunks:
        logger.warning("Aucun document à ingérer n'a été trouvé dans le répertoire.")
        return

    try:
        logger.info("Initialisation du modèle d'embedding...")
        embedding_model = AIFactory.get_embedding_model()
        
        logger.info("Initialisation du gestionnaire de base de données vectorielle...")
        vector_store_manager = VectorStoreManager(embedding_model)
        
        logger.info(f"Ajout de {len(all_chunks)} chunks à la base de données vectorielle...")
        vector_store_manager.add_documents(all_chunks)
        
        logger.info("Ingestion terminée avec succès !")

    except ValueError as e: # Catch API key errors from factory
        logger.error(f"Erreur de configuration: {e}")
        return
    except Exception as e:
        logger.error(f"Une erreur est survenue lors de l'ingestion: {e}")
        return
    
    logger.info("Ingestion terminée avec succès !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingère des documents dans la base de données vectorielle.")
    parser.add_argument("directory", type=str, help="Le chemin vers le répertoire de documents à ingérer.")
    args = parser.parse_args()
    
    ingest_directory(args.directory)
